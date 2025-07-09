# src/dqn/train.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dqn.deep_q_agent import DeepQLearningAgent
from src.shared.envs.grid_environment import GridEnvironment
from src.shared.envs.container_environment import ContainerShipEnv
from src.shared.config import (
    get_training_config,
    DQN_EPISODES,
    MAX_STEPS,
    EVAL_EPISODES,
    get_dqn_model_path,
    EXPORT_PATH,
    SEED
)


class DQNTrainer:
    """Trainer für DQN Agent mit gymnasium Environments"""

    def __init__(self, env_mode: str = "static"):
        self.env_mode = env_mode
        self.training_config = get_training_config()

        # Environment erstellen
        if env_mode == "container":
            self.env = ContainerShipEnv()
            self.env_type = "container"
        else:
            self.env = GridEnvironment(mode=env_mode)
            self.env_type = "grid"

        # Seed setzen
        self.env.seed(SEED)
        self.env.seed_action_space(SEED)

        # Agent erstellen
        self.agent = DeepQLearningAgent()

        print(f"DQN Trainer initialisiert für Modus: {env_mode}")
        print(f"Environment-Typ: {self.env_type}")
        print(f"State Space: {self.env.observation_space}")
        print(f"Action Space: {self.env.action_space}")

    def train(self, episodes: int = None, progress_interval: int = 100) -> Dict[str, Any]:
        """
        Trainiert den DQN Agent.

        Args:
            episodes: Anzahl Trainings-Episoden
            progress_interval: Intervall für Progress-Reports

        Returns:
            Dictionary mit Training-Ergebnissen
        """
        if episodes is None:
            episodes = DQN_EPISODES

        print(f"\nStarte DQN Training...")
        print(f"Modus: {self.env_mode}")
        print(f"Episoden: {episodes}")
        print(f"Max Steps pro Episode: {MAX_STEPS}")
        print()

        # Training-Historie
        episode_rewards = []
        episode_steps = []
        episode_successes = []
        loss_history = []

        for episode in range(episodes):
            # Reset Environment
            obs, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            success = False

            while not done and steps < MAX_STEPS:
                # Wähle Aktion
                action = self.agent.select_action(
                    obs,
                    training=True,
                    env_type=self.env_type
                )

                # Führe Aktion aus
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Check for success (abhängig vom Environment-Typ)
                if terminated:
                    if self.env_type == "grid":
                        # GridEnvironment: Success wenn Ziel erreicht
                        current_pos = self.env.state_to_pos(next_obs)
                        success = (current_pos == self.env.goal_pos)
                    elif self.env_type == "container":
                        # ContainerShipEnv: Success wenn Container abgeliefert
                        success = (self.env.successful_dropoffs > 0)

                # Update Agent
                self.agent.update(
                    obs, action, reward, next_obs, done,
                    env_type=self.env_type
                )

                obs = next_obs
                total_reward += reward
                steps += 1

            # Episode finished
            self.agent.episode_finished(success, steps, total_reward)
            self.agent.decay_exploration()

            # Sammle Metriken
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            episode_successes.append(success)

            if self.agent.losses:
                loss_history.extend(self.agent.losses[-10:])  # Letzte Losses

            # Progress Report
            if (episode + 1) % progress_interval == 0:
                self._print_progress(episode + 1, episodes, episode_successes)

        # Training abgeschlossen
        final_metrics = self.agent.get_metrics()
        success_rate = (sum(episode_successes) / episodes) * 100

        print(f"\nTraining abgeschlossen!")
        print(f"Finale Erfolgsrate: {success_rate:.1f}%")
        self.agent.print_stats()

        # Speichere Modell
        model_path = get_dqn_model_path(self.env_mode)
        self.agent.save_model(model_path)

        return {
            'env_mode': self.env_mode,
            'episodes': episodes,
            'success_rate': success_rate,
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'episode_successes': episode_successes,
            'loss_history': loss_history,
            'final_metrics': final_metrics
        }

    def evaluate(self, episodes: int = None, load_model: bool = True) -> Dict[str, Any]:
        """
        Evaluiert den trainierten Agent.

        Args:
            episodes: Anzahl Evaluations-Episoden
            load_model: Ob gespeichertes Modell geladen werden soll

        Returns:
            Dictionary mit Evaluations-Ergebnissen
        """
        if episodes is None:
            episodes = EVAL_EPISODES

        if load_model:
            model_path = get_dqn_model_path(self.env_mode)
            self.agent.load_model(model_path)

        print(f"\nEvaluiere DQN Agent...")
        print(f"Modus: {self.env_mode}")
        print(f"Episoden: {episodes}")

        results = []
        successes = 0

        for episode in range(episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            success = False

            while not done and steps < MAX_STEPS:
                # Greedy action selection (kein Exploration)
                action = self.agent.select_action(
                    obs,
                    training=False,
                    env_type=self.env_type
                )

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if terminated:
                    if self.env_type == "grid":
                        current_pos = self.env.state_to_pos(obs)
                        success = (current_pos == self.env.goal_pos)
                    elif self.env_type == "container":
                        success = (self.env.successful_dropoffs > 0)

                total_reward += reward
                steps += 1

            results.append({
                'success': success,
                'steps': steps,
                'reward': total_reward
            })

            if success:
                successes += 1

            if episode % 50 == 0:
                print(f"Episode {episode + 1}/{episodes}: "
                      f"{'✓' if success else '✗'} ({steps} Schritte, {total_reward:.1f} Belohnung)")

        # Berechne Statistiken
        success_rate = (successes / episodes) * 100
        avg_steps = np.mean([r['steps'] for r in results])
        std_steps = np.std([r['steps'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        std_reward = np.std([r['reward'] for r in results])

        eval_results = {
            'env_mode': self.env_mode,
            'episodes': episodes,
            'success_rate': success_rate,
            'successes': successes,
            'average_steps': avg_steps,
            'std_steps': std_steps,
            'average_reward': avg_reward,
            'std_reward': std_reward,
            'individual_results': results
        }

        print(f"\nEvaluations-Ergebnisse:")
        print(f"Erfolgsrate: {success_rate:.1f}% ({successes}/{episodes})")
        print(f"Durchschnittliche Schritte: {avg_steps:.1f} ± {std_steps:.1f}")
        print(f"Durchschnittliche Belohnung: {avg_reward:.2f} ± {std_reward:.2f}")

        return eval_results

    def plot_training_results(self, results: Dict[str, Any], save_path: str = None):
        """Plottet Training-Ergebnisse."""
        episodes = results['episodes']
        episode_rewards = results['episode_rewards']
        episode_steps = results['episode_steps']
        episode_successes = results['episode_successes']

        # Gleitender Durchschnitt berechnen
        window_size = max(1, episodes // 20)

        def moving_average(data, window):
            return np.convolve(data, np.ones(window), 'valid') / window

        # Success Rate (gleitender Durchschnitt)
        success_ma = moving_average([int(s) for s in episode_successes], window_size) * 100

        # Reward (gleitender Durchschnitt)
        reward_ma = moving_average(episode_rewards, window_size)

        # Steps (gleitender Durchschnitt)
        steps_ma = moving_average(episode_steps, window_size)

        # Plot erstellen
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'DQN Training Results - {self.env_mode.upper()}', fontsize=14)

        # Success Rate
        x_ma = range(window_size - 1, len(episode_successes))
        axes[0, 0].plot(x_ma, success_ma, 'b-', alpha=0.8)
        axes[0, 0].set_title('Success Rate (Moving Average)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # Episode Rewards
        axes[0, 1].plot(range(len(episode_rewards)), episode_rewards, 'g-', alpha=0.3)
        axes[0, 1].plot(range(window_size - 1, len(episode_rewards)), reward_ma, 'g-', linewidth=2)
        axes[0, 1].set_title('Episode Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].grid(True, alpha=0.3)

        # Episode Steps
        axes[1, 0].plot(range(len(episode_steps)), episode_steps, 'r-', alpha=0.3)
        axes[1, 0].plot(range(window_size - 1, len(episode_steps)), steps_ma, 'r-', linewidth=2)
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)

        # Loss History
        if results['loss_history']:
            loss_ma = moving_average(results['loss_history'], min(50, len(results['loss_history']) // 10))
            axes[1, 1].plot(results['loss_history'], 'orange', alpha=0.3)
            if len(loss_ma) > 0:
                x_loss = range(49, len(results['loss_history']))[:len(loss_ma)]
                axes[1, 1].plot(x_loss, loss_ma, 'orange', linewidth=2)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Loss')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot gespeichert: {save_path}")
        else:
            plt.show()

    def _print_progress(self, episode: int, total_episodes: int, successes: List[bool]):
        """Gibt Training-Fortschritt aus."""
        recent_successes = successes[-100:] if len(successes) >= 100 else successes
        success_rate = (sum(recent_successes) / len(recent_successes)) * 100

        print(f"Episode {episode}/{total_episodes}")
        print(f"  Erfolgsrate (letzte {len(recent_successes)}): {success_rate:.1f}%")
        print(f"  Exploration Rate: {self.agent.exploration_rate:.3f}")
        print(f"  Memory Size: {len(self.agent.memory)}")
        if self.agent.losses:
            recent_loss = np.mean(self.agent.losses[-10:])
            print(f"  Durchschnittlicher Loss: {recent_loss:.4f}")
        print()


def main():
    """Hauptfunktion für Training und Evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description='DQN Training für Schiffsnavigation')
    parser.add_argument('--mode', type=str, default='static',
                        choices=['static', 'random_start', 'random_goal', 'random_obstacles', 'container'],
                        help='Environment-Modus')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Anzahl Training-Episoden')
    parser.add_argument('--eval-only', action='store_true',
                        help='Nur Evaluation (lädt gespeichertes Modell)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Keine Plots erstellen')

    args = parser.parse_args()

    # Trainer erstellen
    trainer = DQNTrainer(env_mode=args.mode)

    if args.eval_only:
        # Nur Evaluation
        eval_results = trainer.evaluate()
    else:
        # Training
        train_results = trainer.train(episodes=args.episodes)

        # Plots erstellen
        if not args.no_plot:
            plot_path = os.path.join(EXPORT_PATH, f'dqn_training_{args.mode}.pdf')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            trainer.plot_training_results(train_results, plot_path)

        # Evaluation nach Training
        print("\n" + "=" * 50)
        eval_results = trainer.evaluate(episodes=100, load_model=False)

    print(f"\nTraining und Evaluation abgeschlossen für Modus: {args.mode}")


if __name__ == "__main__":
    main()