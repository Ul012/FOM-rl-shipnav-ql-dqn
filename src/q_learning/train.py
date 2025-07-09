# train.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os
import numpy as np

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ENV_MODE flexibel setzen (über Umgebungsvariable oder config_ql.py)
ENV_MODE = os.getenv("ENV_MODE", None)
if ENV_MODE is None:
    from src.shared.config import ENV_MODE as CONFIG_ENV_MODE
    ENV_MODE = CONFIG_ENV_MODE

# Lokale Module
from src.shared.config import EPISODES, MAX_STEPS, EPSILON, ALPHA, GAMMA, SEED

# Utils
from utils.common import set_all_seeds, obs_to_state, check_success, setup_export
from utils.environment import initialize_environment
from utils.qlearning import initialize_q_table, select_action, update_q_value, save_q_table
from utils.visualization import create_learning_curve, create_success_curve, create_training_statistics
from utils.reporting import print_training_results

SHOW_VISUALIZATIONS = os.getenv("SHOW_VISUALIZATIONS", "true").lower() == "true"

# ============================================================================
# Hauptfunktion
# ============================================================================

def train_agent(scenario):
    # Seed für Reproduzierbarkeit setzen
    set_all_seeds()

    # Initialisierung
    env, grid_size = initialize_environment(ENV_MODE)
    Q, n_states, n_actions = initialize_q_table(env)
    setup_export()

    # Tracking-Listen
    rewards_per_episode = []
    success_per_episode = []
    steps_per_episode = []

    print(f"Starte Training mit {EPISODES} Episoden...")
    print(f"Episodes: {EPISODES}")
    print("Hyperparameter:")
    print(f"  Lernrate (α): {ALPHA}")
    print(f"  Discount Factor (γ): {GAMMA}")
    print(f"  Epsilon (ε): {EPSILON}")
    print(f"  Seed: {SEED}")

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = obs_to_state(obs, ENV_MODE, grid_size)
        total_reward = 0
        steps = 0
        success = False

        for step in range(MAX_STEPS):
            action = select_action(Q, state, EPSILON, n_actions)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs_to_state(obs, ENV_MODE, grid_size)
            done = terminated or truncated

            update_q_value(Q, state, action, reward, next_state)

            state = next_state
            total_reward += reward
            steps += 1

            if check_success(reward, ENV_MODE):
                success = True

            if done:
                break

        rewards_per_episode.append(total_reward)
        success_per_episode.append(1 if success else 0)
        steps_per_episode.append(steps)

        if (episode + 1) % max(1, EPISODES // 10) == 0:
            recent_episodes = min(100, episode + 1)
            recent_success_rate = np.mean(success_per_episode[-recent_episodes:]) * 100
            print(f"Episode {episode + 1}/{EPISODES}: "
                  f"Reward={total_reward:.2f}, Steps={steps}, "
                  f"Erfolgsrate (letzte {recent_episodes}): {recent_success_rate:.1f}%")

    print_training_results(rewards_per_episode, success_per_episode, steps_per_episode)
    save_q_table(Q, ENV_MODE)

    create_learning_curve(rewards_per_episode, ENV_MODE, show=SHOW_VISUALIZATIONS)
    create_success_curve(success_per_episode, ENV_MODE, show=SHOW_VISUALIZATIONS)
    create_training_statistics(rewards_per_episode, success_per_episode, ENV_MODE, show=SHOW_VISUALIZATIONS)

    # Erweiterte Konsolenausgabe: Reward-Varianz und durchschnittliche Schritte
    reward_mean = np.mean(rewards_per_episode)
    reward_std = np.std(rewards_per_episode)
    reward_var = np.var(rewards_per_episode)
    reward_total = np.sum(rewards_per_episode)

    steps_mean = np.mean(steps_per_episode)
    steps_min = np.min(steps_per_episode)
    steps_max = np.max(steps_per_episode)

    print("\nZusätzliche Metriken:")
    print("  Reward-Statistiken:")
    print(f"    Gesamtreward: {reward_total:.2f}")
    print(f"    Durchschnittlicher Reward: {reward_mean:.2f}")
    print(f"    Standardabweichung: {reward_std:.2f}")
    print(f"    Varianz: {reward_var:.2f}")
    print("  Schritt-Statistiken:")
    print(f"    Durchschnittliche Schritte: {steps_mean:.1f}")
    print(f"    Minimum: {steps_min}")
    print(f"    Maximum: {steps_max}")

    # Speichern der Kurven als .npy
    np.save(f"exports/learning_curve_{scenario}.npy", rewards_per_episode)
    np.save(f"exports/success_curve_{scenario}.npy", success_per_episode)

    return Q, rewards_per_episode, success_per_episode, reward_total

# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    import os
    scenario = os.getenv("ENV_MODE", "static")  # oder eine andere Methode, deinen Szenarionamen zu setzen
    train_agent(scenario)