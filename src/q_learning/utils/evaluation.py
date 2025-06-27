# utils/evaluation.py

import numpy as np
from src.shared.config import REWARDS, LOOP_THRESHOLD

# Klassifikation des Episodenergebnisses
def classify_episode_result(reward, cause, episode_reward, env_mode):
    success = False

    if env_mode == "container":
        if reward == REWARDS["dropoff"]:
            cause = "Ziel erreicht"
            success = True
        elif reward == REWARDS["loop_abort"]:
            cause = "Schleifenabbruch"
        elif reward == REWARDS["obstacle"]:
            cause = "Hindernis-Kollision"
        elif reward == REWARDS["timeout"]:
            cause = "Timeout"
    else:  # Grid-Environment
        if reward == REWARDS["goal"]:
            cause = "Ziel erreicht"
            success = True
        elif reward == REWARDS["loop_abort"] or reward == (REWARDS["loop_abort"] + REWARDS["step"]):
            cause = "Schleifenabbruch"
        elif reward == REWARDS["timeout"] or reward == (REWARDS["timeout"] + REWARDS["step"]):
            cause = "Timeout"
        elif reward == REWARDS["obstacle"] or (reward < 0 and reward != REWARDS["step"]):
            cause = "Hindernis-Kollision"

    return cause, success

# Berechnung der Leistungsmetriken
def calculate_metrics(scenario_results, eval_episodes):
    if scenario_results is None:
        return None

    total = eval_episodes
    return {
        "success_rate": scenario_results["success_count"] / total,
        "timeout_rate": scenario_results["timeout_count"] / total,
        "loop_abort_rate": scenario_results["loop_abort_count"] / total,
        "obstacle_rate": scenario_results["obstacle_count"] / total,
        "avg_reward": np.mean(scenario_results["episode_rewards"]),
        "reward_std": np.std(scenario_results["episode_rewards"]),
        "avg_steps_to_goal": np.mean(scenario_results["steps_to_goal"]) if scenario_results["steps_to_goal"] else None
    }

# Schleifenerkennung für Grid-Environment
def check_loop_detection(visited_states, next_state, env_mode):
    if env_mode != "container":
        visited_states[next_state] = visited_states.get(next_state, 0) + 1
        if visited_states[next_state] >= LOOP_THRESHOLD:
            return True
    return False