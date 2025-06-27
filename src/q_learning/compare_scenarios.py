# compare_scenarios.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Drittanbieter
import numpy as np

# Lokale Module
from shared.config import (EVAL_MAX_STEPS, LOOP_THRESHOLD, EVAL_EPISODES)

# Utils
from utils.common import set_all_seeds, obs_to_state, check_success, setup_export
from utils.environment import initialize_environment_for_scenario
from utils.qlearning import load_q_table
from utils.evaluation import calculate_metrics, check_loop_detection
from utils.visualization import (create_comparison_table, create_success_rate_comparison,
                                create_stacked_failure_chart)

# ============================================================================
# Szenarien-Definition
# ============================================================================

SCENARIOS = {
    "static": {
        "env_mode": "static",
        "q_table_path": "q_table_static.npy",
        "environment": "grid"
    },
    "random_start": {
        "env_mode": "random_start",
        "q_table_path": "q_table_random_start.npy",
        "environment": "grid"
    },
    "random_goal": {
        "env_mode": "random_goal",
        "q_table_path": "q_table_random_goal.npy",
        "environment": "grid"
    },
    "random_obstacles": {
        "env_mode": "random_obstacles",
        "q_table_path": "q_table_random_obstacles.npy",
        "environment": "grid"
    },
    "container": {
        "env_mode": "container",
        "q_table_path": "q_table_container.npy",
        "environment": "container"
    }
}


# ============================================================================
# Evaluation
# ============================================================================

# Evaluation eines einzelnen Szenarios
def evaluate_single_scenario(scenario_name, scenario_config):
    print(f"Evaluiere Szenario: {scenario_name}")

    env, grid_size = initialize_environment_for_scenario(scenario_config)
    Q = load_q_table(scenario_config["env_mode"])

    if Q is None:
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

    for episode in range(EVAL_EPISODES):
        obs, _ = env.reset()
        state = obs_to_state(obs, scenario_config["env_mode"], grid_size)
        episode_reward = 0
        steps = 0
        visited_states = {}

        terminated_by_environment = False
        while steps < EVAL_MAX_STEPS:
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

    total = results["success_count"] + results["timeout_count"] + results["loop_abort_count"] + results["obstacle_count"]
    print(f"  Erfolg: {results['success_count']}, Timeout: {results['timeout_count']}, "
          f"Schleifen: {results['loop_abort_count']}, Hindernisse: {results['obstacle_count']}, Total: {total}")

    return results


# ============================================================================
# Hauptfunktion
# ============================================================================

# Vergleich aller Szenarien
def compare_all_scenarios():
    # Seed für Reproduzierbarkeit setzen
    set_all_seeds()

    print("Starte Szenarien-Vergleich...")
    setup_export()

    all_results = {}
    all_metrics = {}

    for scenario_name, scenario_config in SCENARIOS.items():
        results = evaluate_single_scenario(scenario_name, scenario_config)
        all_results[scenario_name] = results
        all_metrics[scenario_name] = calculate_metrics(results, EVAL_EPISODES)

    create_comparison_table(all_metrics)
    create_success_rate_comparison(all_metrics)
    create_stacked_failure_chart(all_metrics)

    print(f"\n✅ Vergleich abgeschlossen. Parameter aus config.py: EVAL_MAX_STEPS={EVAL_MAX_STEPS}, LOOP_THRESHOLD={LOOP_THRESHOLD}")

    return all_results, all_metrics


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    compare_all_scenarios()