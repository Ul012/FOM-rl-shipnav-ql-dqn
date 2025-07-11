# evaluate_policy.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os
import numpy as np
from collections import defaultdict

# Project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Lokale Module
from src.shared.config import ENV_MODE, MAX_STEPS, LOOP_THRESHOLD, REWARDS, EXPORT_PDF, EXPORT_PATH_QL, SEED
from src.shared.config_utils import get_q_table_path, get_shared_config

# Utils
from utils.common import set_all_seeds, obs_to_state, check_success, setup_export
from utils.environment import initialize_environment
from utils.qlearning import load_q_table
from utils.evaluation import classify_episode_result, check_loop_detection
from utils.visualization import create_success_plot, create_reward_histogram
from utils.reporting import print_evaluation_results


# ============================================================================
# Hauptfunktion
# ============================================================================

def evaluate_policy():
    """Evaluation der trainierten Policy über mehrere Episoden"""

    # Konfiguration laden - ANGEPASST
    shared_config = get_shared_config()
    eval_episodes = shared_config['eval_episodes']
    eval_max_steps = shared_config['eval_max_steps']

    print("POLICY EVALUATION")
    print("=" * 50)
    print(f"Szenario: {ENV_MODE}")
    print(f"Eval Episodes: {eval_episodes}")
    print(f"Max Steps pro Episode: {eval_max_steps}")
    print(f"Seed: {shared_config['seed']}")
    print("=" * 50)

    # Seed für Reproduzierbarkeit setzen
    set_all_seeds()

    # Initialisierung
    env, grid_size = initialize_environment(ENV_MODE)
    q_table_path = get_q_table_path(ENV_MODE)
    Q = load_q_table(q_table_path)
    setup_export()

    if Q is None:
        print(f"FEHLER: Q-Tabelle nicht gefunden: {q_table_path}")
        print("Bitte führen Sie zuerst das Training aus.")
        return

    results_cause = defaultdict(int)
    results_solved = defaultdict(int)
    rewards_all = []

    print(f"\nStarte Evaluation...")

    for episode in range(eval_episodes):
        obs, _ = env.reset()
        state = obs_to_state(obs, ENV_MODE, grid_size)
        episode_reward = 0
        visited_states = {}
        steps = 0
        cause = "Timeout"
        goal_reached = False
        loop_detected = False

        # Episode durchführen - ANGEPASST: eval_max_steps verwenden
        for step in range(eval_max_steps):
            # Greedy Policy (kein Epsilon, da Evaluation)
            action = np.argmax(Q[state])
            obs, reward, terminated, _, _ = env.step(action)
            next_state = obs_to_state(obs, ENV_MODE, grid_size)
            episode_reward += reward
            steps += 1

            # Schleifenerkennung für Grid-Environment
            if check_loop_detection(visited_states, next_state, ENV_MODE):
                cause = "Schleifenabbruch"
                loop_detected = True
                episode_reward += REWARDS["loop_abort"]
                break

            # Erfolg erkennen
            if ENV_MODE == "container":
                if reward == REWARDS["dropoff"]:
                    goal_reached = True
                    cause = "Ziel erreicht"
                elif reward == REWARDS["loop_abort"]:
                    loop_detected = True
                    cause = "Schleifenabbruch"
            else:
                if reward == REWARDS["goal"]:
                    goal_reached = True
                    cause = "Ziel erreicht"

            state = next_state

            if terminated:
                # Klassifizierung basierend auf letztem Reward
                if not goal_reached and not loop_detected:
                    cause, _ = classify_episode_result(reward, cause, episode_reward, ENV_MODE)
                break

        # Timeout-Check - ANGEPASST: eval_max_steps verwenden
        if not terminated and steps >= eval_max_steps:
            cause = "Timeout"
            episode_reward += REWARDS["timeout"]

        # Alternative Klassifizierung basierend auf Gesamt-Reward
        if cause == "Timeout" and episode_reward > 0:
            if ENV_MODE != "container" and episode_reward >= REWARDS["goal"]:
                cause = "Ziel erreicht"

        # Erfolg bestimmen
        success = (cause == "Ziel erreicht")

        # Statistiken aktualisieren
        results_cause[cause] += 1
        results_solved["solved episode" if success else "failed episode"] += 1
        rewards_all.append(episode_reward)

        # Progress Report
        if (episode + 1) % max(1, eval_episodes // 10) == 0:
            current_success_rate = results_solved["solved episode"] / (episode + 1) * 100
            print(f"Episode {episode + 1}/{eval_episodes}: "
                  f"Erfolgsrate: {current_success_rate:.1f}%, "
                  f"Letzter Reward: {episode_reward:.1f}")

    # Ergebnisse ausgeben
    print("\n" + "=" * 50)
    print("EVALUATION ABGESCHLOSSEN")
    print_evaluation_results(results_cause, results_solved, rewards_all, eval_episodes, ENV_MODE)

    # Visualisierungen erstellen
    create_success_plot(results_solved, ENV_MODE)
    create_reward_histogram(rewards_all, ENV_MODE)

    print("=" * 50)
    return results_cause, results_solved, rewards_all


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    evaluate_policy()