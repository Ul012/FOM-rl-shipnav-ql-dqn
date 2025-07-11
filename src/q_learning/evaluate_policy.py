# evaluate_policy.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Drittanbieter
import numpy as np
from collections import defaultdict

# Lokale Module
from shared.config import (ENV_MODE, EPISODES, MAX_STEPS, LOOP_THRESHOLD, REWARDS, EXPORT_PDF, EXPORT_PATH_QL, SEED)
from shared_config_utils import get_q_table_path

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

# Evaluation der trainierten Policy über mehrere Episoden
def evaluate_policy():
    # Seed für Reproduzierbarkeit setzen
    set_all_seeds()

    # Initialisierung
    env, grid_size = initialize_environment(ENV_MODE)
    q_table_path = get_q_table_path(ENV_MODE)
    Q = load_q_table(q_table_path)
    setup_export()

    if Q is None:
        return

    results_cause = defaultdict(int)
    results_solved = defaultdict(int)
    rewards_all = []

    print(f"Starte Evaluation mit {EPISODES} Episoden...")

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = obs_to_state(obs, ENV_MODE, grid_size)
        episode_reward = 0
        visited_states = {}
        steps = 0
        cause = "Timeout"
        goal_reached = False
        loop_detected = False

        # Episode durchführen
        for step in range(MAX_STEPS):
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

        # Timeout-Check
        if not terminated and steps >= MAX_STEPS:
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

    # Ergebnisse ausgeben
    print_evaluation_results(results_cause, results_solved, rewards_all, EPISODES, ENV_MODE)

    # Visualisierungen erstellen
    create_success_plot(results_solved, ENV_MODE)
    create_reward_histogram(rewards_all, ENV_MODE)


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    evaluate_policy()