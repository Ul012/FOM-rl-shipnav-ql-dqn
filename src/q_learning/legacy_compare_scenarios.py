# legacy_compare_scenarios.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Drittanbieter
import numpy as np

# Lokale Module - ANGEPASST für vereinheitlichte Config
from src.shared.config import LOOP_THRESHOLD
from src.shared.config_utils import get_shared_config, get_q_table_path

# Utils
from utils.common import set_all_seeds, obs_to_state, check_success, setup_export
from utils.environment import initialize_environment_for_scenario
from utils.qlearning import load_q_table
from utils.evaluation import calculate_metrics, check_loop_detection
from utils.visualization import (create_comparison_table, create_success_rate_comparison,
                                 create_stacked_failure_chart)

# ============================================================================
# Szenarien-Definition - ANGEPASST
# ============================================================================

SCENARIOS = {
    "static": {
        "env_mode": "static",
        "environment": "grid"
    },
    "random_start": {
        "env_mode": "random_start",
        "environment": "grid"
    },
    "random_goal": {
        "env_mode": "random_goal",
        "environment": "grid"
    },
    "random_obstacles": {
        "env_mode": "random_obstacles",
        "environment": "grid"
    },
    "container": {
        "env_mode": "container",
        "environment": "container"
    }
}


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_single_scenario(scenario_name, scenario_config):
    """Evaluation eines einzelnen Szenarios - ANGEPASST"""

    # Konfiguration laden - ANGEPASST
    shared_config = get_shared_config()
    eval_episodes = shared_config['eval_episodes']
    eval_max_steps = shared_config['eval_max_steps']

    print(f"Evaluiere Szenario: {scenario_name}")
    print(f"  Episodes: {eval_episodes}, Max Steps: {eval_max_steps}")

    env, grid_size = initialize_environment_for_scenario(scenario_config)

    # Q-Table Path über config_utils - ANGEPASST
    Q = load_q_table(get_q_table_path(scenario_config["env_mode"]))

    if Q is None:
        print(f"  ❌ Q-Tabelle für {scenario_name} nicht gefunden!")
        return None

    results = {
        "success_count": 0,
        "timeout_count": 0,
        "loop_abort_count": 0,
        "obstacle_count": 0,
        "episode_rewards": [],
        "steps_to_goal": [],
        "success_per_episode": []
    }

    for episode in range(eval_episodes):
        obs, _ = env.reset()
        state = obs_to_state(obs, scenario_config["env_mode"], grid_size)
        episode_reward = 0
        steps = 0
        visited_states = {}

        terminated_by_environment = False

        # Episode durchführen - ANGEPASST: eval_max_steps verwenden
        while steps < eval_max_steps:
            # Greedy Policy (kein Epsilon, da Evaluation)
            action = np.argmax(Q[state])
            obs, reward, terminated, _, _ = env.step(action)
            next_state = obs_to_state(obs, scenario_config["env_mode"], grid_size)
            episode_reward += reward
            steps += 1

            if check_success(reward, scenario_config["env_mode"]):
                results["success_count"] += 1
                results["steps_to_goal"].append(steps)
                break

            if check_loop_detection(visited_states, next_state, scenario_config["env_mode"]):
                results["loop_abort_count"] += 1
                break

            if terminated:
                results["obstacle_count"] += 1
                terminated_by_environment = True
                break

            state = next_state
        else:
            if not terminated_by_environment:
                results["timeout_count"] += 1

        results["episode_rewards"].append(episode_reward)
        results["success_per_episode"].append(
            1 if results["success_count"] > len(results["success_per_episode"]) else 0)

    total = results["success_count"] + results["timeout_count"] + results["loop_abort_count"] + results[
        "obstacle_count"]
    success_rate = (results["success_count"] / eval_episodes) * 100 if eval_episodes > 0 else 0

    print(f"  ✅ Erfolg: {results['success_count']}/{eval_episodes} ({success_rate:.1f}%)")
    print(f"     Timeout: {results['timeout_count']}, Schleifen: {results['loop_abort_count']}, "
          f"Hindernisse: {results['obstacle_count']}")

    return results


# ============================================================================
# Hauptfunktion
# ============================================================================

def compare_all_scenarios():
    """Vergleich aller Szenarien - ANGEPASST"""

    # Konfiguration laden - ANGEPASST
    shared_config = get_shared_config()
    eval_episodes = shared_config['eval_episodes']
    eval_max_steps = shared_config['eval_max_steps']

    print("SZENARIEN-VERGLEICH")
    print("=" * 60)
    print(f"Evaluation Episodes: {eval_episodes}")
    print(f"Max Steps pro Episode: {eval_max_steps}")
    print(f"Loop Threshold: {LOOP_THRESHOLD}")
    print(f"Seed: {shared_config['seed']}")
    print("=" * 60)

    # Seed für Reproduzierbarkeit setzen
    set_all_seeds()
    setup_export()

    all_results = {}
    all_metrics = {}

    print("\nStarte Evaluation aller Szenarien...")

    for scenario_name, scenario_config in SCENARIOS.items():
        results = evaluate_single_scenario(scenario_name, scenario_config)
        if results is not None:
            all_results[scenario_name] = results
            all_metrics[scenario_name] = calculate_metrics(results, eval_episodes)
        else:
            print(f"  ⚠️  Überspringe {scenario_name} (Q-Tabelle nicht verfügbar)")

    if not all_metrics:
        print("\n❌ Keine Q-Tabellen für Vergleich gefunden!")
        print("Bitte führen Sie zuerst das Training für alle Szenarien aus.")
        return None, None

    print(f"\n{'=' * 60}")
    print("ERSTELLUNG DER VERGLEICHSGRAFIKEN")
    print(f"{'=' * 60}")

    # Visualisierungen erstellen
    create_comparison_table(all_metrics)
    create_success_rate_comparison(all_metrics)
    create_stacked_failure_chart(all_metrics)

    print(f"\n✅ Vergleich abgeschlossen!")
    print(f"Parameter: EVAL_MAX_STEPS={eval_max_steps}, LOOP_THRESHOLD={LOOP_THRESHOLD}")
    print(f"Erfolgreich evaluierte Szenarien: {list(all_metrics.keys())}")

    return all_results, all_metrics


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    compare_all_scenarios()