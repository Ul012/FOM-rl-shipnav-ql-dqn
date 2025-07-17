# src/comparison/compare_algorithms_2x3.py

import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.shared.config import (
    SEED, SETUP_NAME, EPISODES, EVAL_EPISODES, EVAL_MAX_STEPS, EXPORT_PATH_COMP, EXPORT_PATH_QL, EXPORT_PATH_DQN,
    GAMMA, QL_ALPHA, QL_EPSILON_FIXED, DQN_EPSILON_FIXED, DQN_LEARNING_RATE, EPSILON_START, EPSILON_END, EPSILON_DECAY
)
from src.shared.config_utils import get_dqn_model_path, get_q_table_path, get_export_path


@dataclass
class ComparisonConfig:
    base_seed: int = SEED
    eval_episodes: int = EVAL_EPISODES
    max_steps_per_episode: int = EVAL_MAX_STEPS
    scenarios: List[str] = None

    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = ['static', 'random_start', 'random_goal', 'random_obstacles', 'container']

    def get_seed_for_run(self, run: int) -> int:
        return self.base_seed + run


@dataclass
class EvaluationResult:
    success_rate: float
    average_steps: float
    average_reward: float
    duration: float
    additional_metrics: Dict[str, Any] = None


class AlgorithmEvaluator(ABC):
    def __init__(self, config: ComparisonConfig):
        self.config = config

    @abstractmethod
    def evaluate(self, scenario: str, episodes: int, run: int) -> EvaluationResult:
        pass

    @abstractmethod
    def is_model_available(self, scenario: str) -> bool:
        pass


class QLearningEvaluator(AlgorithmEvaluator):
    def __init__(self, config: ComparisonConfig):
        super().__init__(config)
        sys.path.append(os.path.join(os.path.dirname(__file__), "../q_learning"))
        from utils.common import set_all_seeds, obs_to_state, check_success
        from utils.environment import initialize_environment_for_scenario
        from utils.qlearning import load_q_table
        from utils.evaluation import check_loop_detection

        self.set_all_seeds = set_all_seeds
        self.obs_to_state = obs_to_state
        self.check_success = check_success
        self.initialize_environment_for_scenario = initialize_environment_for_scenario
        self.load_q_table = load_q_table
        self.check_loop_detection = check_loop_detection

    def is_model_available(self, scenario: str) -> bool:
        q_table_path = get_q_table_path(scenario)
        full_path = os.path.join(os.path.dirname(__file__), "../q_learning", q_table_path)
        return os.path.exists(full_path)

    def evaluate(self, scenario: str, episodes: int, run: int) -> EvaluationResult:
        start_time = time.time()
        seed = self.config.get_seed_for_run(run)
        self.set_all_seeds()
        np.random.seed(seed)

        scenario_config = {
            "env_mode": scenario,
            "q_table_path": get_q_table_path(scenario),
            "environment": "container" if scenario == "container" else "grid"
        }

        env, grid_size = self.initialize_environment_for_scenario(scenario_config)
        q_table_path = get_q_table_path(scenario)
        q_table_full_path = os.path.join(os.path.dirname(__file__), "../q_learning", q_table_path)
        Q = self.load_q_table(q_table_full_path)

        if Q is None:
            raise FileNotFoundError(f"Q-Table für Szenario '{scenario}' nicht gefunden")

        results = self._run_evaluation_episodes(env, Q, grid_size, scenario, episodes)
        duration = time.time() - start_time

        return EvaluationResult(
            success_rate=results['success_rate'],
            average_steps=results['average_steps'],
            average_reward=results['average_reward'],
            duration=duration
        )

    def _run_evaluation_episodes(self, env, Q, grid_size, scenario, episodes):
        results = {"success_count": 0, "episode_rewards": [], "steps_to_goal": []}
        max_steps = self.config.max_steps_per_episode

        for episode in range(episodes):
            obs, _ = env.reset()
            state = self.obs_to_state(obs, scenario, grid_size)
            episode_reward = 0
            steps = 0
            visited_states = {}

            while steps < max_steps:
                action = np.argmax(Q[state])
                obs, reward, terminated, _, _ = env.step(action)
                next_state = self.obs_to_state(obs, scenario, grid_size)
                episode_reward += reward
                steps += 1

                if self.check_success(reward, scenario):
                    results["success_count"] += 1
                    results["steps_to_goal"].append(steps)
                    break

                if self.check_loop_detection(visited_states, next_state, scenario) or terminated:
                    break

                state = next_state

            results["episode_rewards"].append(episode_reward)

        success_rate = (results["success_count"] / episodes) * 100
        average_steps = np.mean(results["steps_to_goal"]) if results["steps_to_goal"] else max_steps
        average_reward = np.mean(results["episode_rewards"])

        return {
            'success_rate': success_rate,
            'average_steps': average_steps,
            'average_reward': average_reward
        }


class DQNEvaluator(AlgorithmEvaluator):
    def __init__(self, config: ComparisonConfig):
        super().__init__(config)
        from src.dqn.train import DQNTrainer
        self.DQNTrainer = DQNTrainer

    def is_model_available(self, scenario: str) -> bool:
        model_path = get_dqn_model_path(scenario)
        full_path = os.path.join(os.path.dirname(__file__), "../dqn", model_path)
        return os.path.exists(full_path)

    def evaluate(self, scenario: str, episodes: int, run: int) -> EvaluationResult:
        start_time = time.time()
        seed = self.config.get_seed_for_run(run)
        np.random.seed(seed)

        trainer = self.DQNTrainer(env_mode=scenario)
        eval_results = trainer.evaluate(episodes=episodes, load_model=True)
        duration = time.time() - start_time

        return EvaluationResult(
            success_rate=eval_results['success_rate'],
            average_steps=eval_results['average_steps'],
            average_reward=eval_results['average_reward'],
            duration=duration
        )


class AlgorithmComparison:
    def __init__(self, config: ComparisonConfig = None):
        self.config = config or ComparisonConfig()
        self.evaluators = {
            "Q-Learning": QLearningEvaluator(self.config),
            "DQN": DQNEvaluator(self.config)
        }
        self.colors = {'Q-Learning': '#1f77b4', 'DQN': '#ff7f0e'}

    def run_comparison(self, num_runs: int = 3) -> List[Dict]:
        print("🎯 ALGORITHMUS-VERGLEICH (basierend auf train_all_scenarios)")

        # Absolute Pfade basierend auf aktuellem SETUP_NAME
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        ql_csv = os.path.join(project_root, "src", "q_learning", "exports", SETUP_NAME, "evaluation_summary.csv")
        dqn_csv = os.path.join(project_root, "src", "dqn", "exports", SETUP_NAME, "dqn_all_scenarios_summary.csv")

        comparison_data = []

        # ✅ DEBUG: Was ist in den CSVs?
        print(f"Q-Learning CSV: {ql_csv}")
        print(f"DQN CSV: {dqn_csv}")

        if os.path.exists(ql_csv):
            ql_df = pd.read_csv(ql_csv)
            print(f"✅ Q-Learning Spalten: {list(ql_df.columns)}")
            print(f"   Erste Zeile: {ql_df.iloc[0].to_dict()}")

            for _, row in ql_df.iterrows():
                comparison_data.append({
                    'algorithm': 'Q-Learning',
                    'scenario': row['Szenario'],
                    'run': 1,
                    'success_rate': row['Success Rate (%)'],
                    'avg_steps': row['Avg. Steps'],
                    'avg_reward': row['Avg. Reward']
                })
        else:
            print(f"❌ Q-Learning CSV nicht gefunden!")

        if os.path.exists(dqn_csv):
            dqn_df = pd.read_csv(dqn_csv)
            print(f"✅ DQN Spalten: {list(dqn_df.columns)}")
            print(f"   Erste Zeile: {dqn_df.iloc[0].to_dict()}")

            for _, row in dqn_df.iterrows():
                comparison_data.append({
                    'algorithm': 'DQN',
                    'scenario': row['scenario'],
                    'run': 1,
                    'success_rate': row['eval_success_rate'],  # ✅ Korrekte Spalte
                    'avg_steps': row['eval_avg_steps'],  # ✅ Korrekte Spalte
                    'avg_reward': row['eval_avg_reward']  # ✅ Korrekte Spalte
                })
        else:
            print(f"❌ DQN CSV nicht gefunden!")

        print(f"📊 Comparison Data: {len(comparison_data)} Einträge")
        if comparison_data:
            print(f"   Beispiel: {comparison_data[0]}")
        else:
            print("❌ Keine Daten gefunden - Abbruch!")
            return []

        self._create_visualization(comparison_data)
        self._save_results(comparison_data)
        return comparison_data

    def _create_visualization(self, comparison_data: List[Dict]):
        df = pd.DataFrame(comparison_data)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Algorithm Comparison: Q-Learning vs Deep Q-Learning', fontsize=16)

        scenarios = self.config.scenarios
        algorithms = ['Q-Learning', 'DQN']
        x = np.arange(len(scenarios))
        width = 0.35

        # Success Rate
        for i, algorithm in enumerate(algorithms):
            means = []
            stds = []
            for scenario in scenarios:
                data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
                means.append(data['success_rate'].mean() if len(data) > 0 else 0)
                stds.append(data['success_rate'].std() if len(data) > 0 else 0)

            axes[0, 0].bar(x + i * width - width / 2, means, width,
                           yerr=stds, capsize=3, alpha=0.8,
                           color=self.colors[algorithm], label=algorithm)

        axes[0, 0].set_title('Erfolgsrate')
        axes[0, 0].set_ylabel('Erfolgsrate (%)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scenarios, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Average Steps
        for i, algorithm in enumerate(algorithms):
            means = []
            stds = []
            for scenario in scenarios:
                data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
                means.append(data['avg_steps'].mean() if len(data) > 0 else 0)
                stds.append(data['avg_steps'].std() if len(data) > 0 else 0)

            axes[0, 1].bar(x + i * width - width / 2, means, width,
                           yerr=stds, capsize=3, alpha=0.8,
                           color=self.colors[algorithm], label=algorithm)

        axes[0, 1].set_title('Durchschnittliche Schritte')
        axes[0, 1].set_ylabel('Schritte')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scenarios, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Average Reward
        for i, algorithm in enumerate(algorithms):
            means = []
            stds = []
            for scenario in scenarios:
                data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
                means.append(data['avg_reward'].mean() if len(data) > 0 else 0)
                stds.append(data['avg_reward'].std() if len(data) > 0 else 0)

            axes[0, 2].bar(x + i * width - width / 2, means, width,
                           yerr=stds, capsize=3, alpha=0.8,
                           color=self.colors[algorithm], label=algorithm)

        axes[0, 2].set_title('Durchschnittliche Belohnung')
        axes[0, 2].set_ylabel('Belohnung')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(scenarios, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Box Plots
        for i, (metric, title) in enumerate([('success_rate', 'Erfolgsrate Verteilung'),
                                           ('avg_steps', 'Schritte Verteilung'),
                                           ('avg_reward', 'Belohnung Verteilung')]):
            data = [df[df['algorithm'] == algo][metric].values for algo in algorithms]
            bp = axes[1, i].boxplot(data, tick_labels=algorithms, patch_artist=True)
            for patch, color in zip(bp['boxes'], [self.colors[algo] for algo in algorithms]):
                patch.set_facecolor(color)
            axes[1, i].set_title(title)
            axes[1, i].set_ylabel(metric.replace('_', ' ').title())
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(get_export_path(EXPORT_PATH_COMP), f'{SETUP_NAME}_algorithm_comparison_2x3_Visual0.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2x3 Vergleich gespeichert: {save_path}")
        plt.close()

    def _save_results(self, comparison_data: List[Dict]):
        df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(get_export_path(EXPORT_PATH_COMP), 'algorithm_comparison_2x3.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Daten gespeichert: {csv_path}")


def main():
    comparison = AlgorithmComparison()
    comparison.run_comparison(num_runs=3)


if __name__ == "__main__":
    main()