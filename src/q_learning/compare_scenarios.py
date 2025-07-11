# src/q_learning/compare_scenarios.py - Minimal Clean Code

import sys
import os
import numpy as np

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.config import LOOP_THRESHOLD
from src.shared.config_utils import get_shared_config, get_q_table_path
from utils.common import set_all_seeds, obs_to_state, check_success, setup_export
from utils.environment import initialize_environment_for_scenario
from utils.qlearning import load_q_table
from utils.evaluation import calculate_metrics, check_loop_detection
from utils.visualization import (create_comparison_table, create_success_rate_comparison,
                                 create_stacked_failure_chart)

# ============================================================================
# Konstanten - eliminiert Magic Strings
# ============================================================================

RESULT_KEYS = {
    'SUCCESS_COUNT': 'success_count',
    'TIMEOUT_COUNT': 'timeout_count',
    'LOOP_ABORT_COUNT': 'loop_abort_count',
    'OBSTACLE_COUNT': 'obstacle_count',
    'EPISODE_REWARDS': 'episode_rewards',
    'STEPS_TO_GOAL': 'steps_to_goal',
    'SUCCESS_PER_EPISODE': 'success_per_episode'
}

SCENARIOS = {
    "static": {"env_mode": "static", "environment": "grid"},
    "random_start": {"env_mode": "random_start", "environment": "grid"},
    "random_goal": {"env_mode": "random_goal", "environment": "grid"},
    "random_obstacles": {"env_mode": "random_obstacles", "environment": "grid"},
    "container": {"env_mode": "container", "environment": "container"}
}


# ============================================================================
# Hilfsfunktionen - geteilte Logik extrahiert
# ============================================================================

def load_evaluation_config():
    """Lädt und gibt Evaluation-Konfiguration zurück"""
    shared_config = get_shared_config()
    return {
        'episodes': shared_config['eval_episodes'],
        'max_steps': shared_config['eval_max_steps'],
        'seed': shared_config['seed']
    }


def create_empty_results():
    """Erstellt leere Ergebnis-Struktur"""
    return {
        RESULT_KEYS['SUCCESS_COUNT']: 0,
        RESULT_KEYS['TIMEOUT_COUNT']: 0,
        RESULT_KEYS['LOOP_ABORT_COUNT']: 0,
        RESULT_KEYS['OBSTACLE_COUNT']: 0,
        RESULT_KEYS['EPISODE_REWARDS']: [],
        RESULT_KEYS['STEPS_TO_GOAL']: [],
        RESULT_KEYS['SUCCESS_PER_EPISODE']: []
    }


def print_config_header(config):
    """Gibt Konfigurations-Header aus"""
    print("SZENARIEN-VERGLEICH")
    print("=" * 60)
    print(f"Evaluation Episodes: {config['episodes']}")
    print(f"Max Steps pro Episode: {config['max_steps']}")
    print(f"Loop Threshold: {LOOP_THRESHOLD}")
    print(f"Seed: {config['seed']}")
    print("=" * 60)


def print_scenario_start(scenario_name, config):
    """Gibt Szenario-Start aus"""
    print(f"Evaluiere Szenario: {scenario_name}")
    print(f"  Episodes: {config['episodes']}, Max Steps: {config['max_steps']}")


def print_scenario_result(scenario_name, results, config):
    """Gibt Szenario-Ergebnis aus"""
    total = config['episodes']
    success_rate = (results[RESULT_KEYS['SUCCESS_COUNT']] / total) * 100

    print(f"  ✅ Erfolg: {results[RESULT_KEYS['SUCCESS_COUNT']]}/{total} ({success_rate:.1f}%)")
    print(f"     Timeout: {results[RESULT_KEYS['TIMEOUT_COUNT']]}, "
          f"Schleifen: {results[RESULT_KEYS['LOOP_ABORT_COUNT']]}, "
          f"Hindernisse: {results[RESULT_KEYS['OBSTACLE_COUNT']]}")


# ============================================================================
# Haupt-Evaluation - aufgeteilt in kleinere Funktionen
# ============================================================================

def setup_scenario(scenario_config):
    """Setzt Environment und Q-Table für Szenario auf"""
    env, grid_size = initialize_environment_for_scenario(scenario_config)
    q_table = load_q_table(get_q_table_path(scenario_config["env_mode"]))

    if q_table is None:
        raise FileNotFoundError(f"Q-Table nicht gefunden")

    return env, grid_size, q_table


def run_single_episode(env, q_table, grid_size, scenario_config, max_steps):
    """Führt eine einzelne Episode aus - FOKUSSIERT auf Kernlogik"""
    obs, _ = env.reset()
    state = obs_to_state(obs, scenario_config["env_mode"], grid_size)

    episode_reward = 0
    steps = 0
    visited_states = {}

    for step in range(max_steps):
        # Greedy Policy
        action = np.argmax(q_table[state])
        obs, reward, terminated, _, _ = env.step(action)
        next_state = obs_to_state(obs, scenario_config["env_mode"], grid_size)

        episode_reward += reward
        steps += 1

        # Success Check
        if check_success(reward, scenario_config["env_mode"]):
            return True, steps, episode_reward, 'success'

        # Loop Check
        if check_loop_detection(visited_states, next_state, scenario_config["env_mode"]):
            return False, steps, episode_reward, 'loop'

        # Termination Check
        if terminated:
            return False, steps, episode_reward, 'obstacle'

        state = next_state

    # Timeout
    return False, steps, episode_reward, 'timeout'


def evaluate_single_scenario(scenario_name, scenario_config, config):
    """Evaluiert ein Szenario - REDUZIERT auf Essentials"""
    print_scenario_start(scenario_name, config)

    try:
        env, grid_size, q_table = setup_scenario(scenario_config)
    except FileNotFoundError:
        print(f"  ❌ Q-Tabelle für {scenario_name} nicht gefunden!")
        return None

    results = create_empty_results()

    # Episoden durchführen
    for episode in range(config['episodes']):
        success, steps, reward, reason = run_single_episode(
            env, q_table, grid_size, scenario_config, config['max_steps']
        )

        # Ergebnisse sammeln
        results[RESULT_KEYS['EPISODE_REWARDS']].append(reward)

        if success:
            results[RESULT_KEYS['SUCCESS_COUNT']] += 1
            results[RESULT_KEYS['STEPS_TO_GOAL']].append(steps)
            results[RESULT_KEYS['SUCCESS_PER_EPISODE']].append(1)
        else:
            results[RESULT_KEYS['SUCCESS_PER_EPISODE']].append(0)

            # Kategorisiere Fehlschläge
            if reason == 'timeout':
                results[RESULT_KEYS['TIMEOUT_COUNT']] += 1
            elif reason == 'loop':
                results[RESULT_KEYS['LOOP_ABORT_COUNT']] += 1
            elif reason == 'obstacle':
                results[RESULT_KEYS['OBSTACLE_COUNT']] += 1

    print_scenario_result(scenario_name, results, config)
    return results


# ============================================================================
# Hauptfunktion - VEREINFACHT
# ============================================================================

def compare_all_scenarios():
    """Hauptfunktion für Szenarien-Vergleich"""
    # Setup
    config = load_evaluation_config()
    set_all_seeds()
    setup_export()

    print_config_header(config)
    print("\nStarte Evaluation aller Szenarien...")

    # Evaluation
    all_results = {}
    all_metrics = {}

    for scenario_name, scenario_config in SCENARIOS.items():
        results = evaluate_single_scenario(scenario_name, scenario_config, config)
        if results is not None:
            all_results[scenario_name] = results
            all_metrics[scenario_name] = calculate_metrics(results, config['episodes'])
        else:
            print(f"  ⚠️  Überspringe {scenario_name} (Q-Tabelle nicht verfügbar)")

    # Validierung
    if not all_metrics:
        print("\n❌ Keine Q-Tabellen für Vergleich gefunden!")
        print("Bitte führen Sie zuerst das Training für alle Szenarien aus.")
        return None, None

    # Visualisierung
    print(f"\n{'=' * 60}")
    print("ERSTELLUNG DER VERGLEICHSGRAFIKEN")
    print(f"{'=' * 60}")

    create_comparison_table(all_metrics)
    create_success_rate_comparison(all_metrics)
    create_stacked_failure_chart(all_metrics)

    # Summary
    print(f"\n✅ Vergleich abgeschlossen!")
    print(f"Parameter: EVAL_MAX_STEPS={config['max_steps']}, LOOP_THRESHOLD={LOOP_THRESHOLD}")
    print(f"Erfolgreich evaluierte Szenarien: {list(all_metrics.keys())}")

    return all_results, all_metrics


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    compare_all_scenarios()