# src/dqn/compare_scenarios.py - Clean Code

import sys
import os
import numpy as np

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.shared.config import LOOP_THRESHOLD, EVAL_MAX_STEPS
from src.shared.config_utils import get_shared_config, get_dqn_model_path
from src.dqn.train import DQNTrainer

# Utils für Setup (aus Q-Learning, aber funktioniert auch für DQN)
sys.path.append(os.path.join(os.path.dirname(__file__), "../q_learning"))
from utils.common import set_all_seeds, setup_export

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
    print("DQN SZENARIEN-VERGLEICH")
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
# DQN-spezifische Evaluation
# ============================================================================

def convert_dqn_to_qlearning_format(dqn_results, total_episodes, max_steps):
    """
    Konvertiert DQN-Evaluationsergebnisse in Q-Learning-kompatibles Format.
    EXAKT wie Q-Learning evaluate_single_scenario() Ausgabe.
    """
    success_count = dqn_results.get('successes', 0)
    individual_results = dqn_results.get('individual_results', [])

    # Initialisiere Zähler EXAKT wie Q-Learning
    results = create_empty_results()
    results[RESULT_KEYS['SUCCESS_COUNT']] = success_count

    # Verarbeite individuelle Ergebnisse
    for i, result in enumerate(individual_results):
        episode_reward = result.get('reward', 0)
        episode_steps = result.get('steps', max_steps)
        episode_success = result.get('success', False)

        results[RESULT_KEYS['EPISODE_REWARDS']].append(episode_reward)

        if episode_success:
            results[RESULT_KEYS['STEPS_TO_GOAL']].append(episode_steps)
            results[RESULT_KEYS['SUCCESS_PER_EPISODE']].append(1)
        else:
            results[RESULT_KEYS['SUCCESS_PER_EPISODE']].append(0)
            # Klassifiziere Fehlschläge (DQN unterscheidet nicht, daher als timeout)
            if episode_steps >= max_steps:
                results[RESULT_KEYS['TIMEOUT_COUNT']] += 1
            else:
                # Könnte Hindernis oder andere Terminierung sein
                results[RESULT_KEYS['OBSTACLE_COUNT']] += 1

    # Stelle sicher, dass alle Episoden gezählt sind
    total_failures = total_episodes - results[RESULT_KEYS['SUCCESS_COUNT']]
    counted_failures = results[RESULT_KEYS['TIMEOUT_COUNT']] + results[RESULT_KEYS['LOOP_ABORT_COUNT']] + results[
        RESULT_KEYS['OBSTACLE_COUNT']]

    if counted_failures != total_failures:
        # Anpassung: Fehlende Fehlschläge als timeout klassifizieren
        results[RESULT_KEYS['TIMEOUT_COUNT']] = total_failures - results[RESULT_KEYS['LOOP_ABORT_COUNT']] - results[
            RESULT_KEYS['OBSTACLE_COUNT']]

    return results


def evaluate_single_scenario(scenario_name, scenario_config, config):
    """Evaluiert ein DQN-Szenario - ANGEPASST für DQN"""
    print_scenario_start(scenario_name, config)

    # DQN Trainer erstellen
    try:
        trainer = DQNTrainer(env_mode=scenario_config["env_mode"])
    except Exception as e:
        print(f"  ❌ Fehler beim Erstellen des DQN Trainers für {scenario_name}: {e}")
        return None

    # Modell-Pfad prüfen
    model_path = get_dqn_model_path(scenario_config["env_mode"])

    if not os.path.exists(model_path):
        print(f"  ❌ DQN-Modell für {scenario_name} nicht gefunden: {model_path}")
        return None

    # DQN Evaluation durchführen mit EVAL_MAX_STEPS
    try:
        # Temporär MAX_STEPS für Evaluation überschreiben
        original_max_steps = trainer.env.max_steps if hasattr(trainer.env, 'max_steps') else None
        if hasattr(trainer.env, 'max_steps'):
            trainer.env.max_steps = config['max_steps']

        # Evaluate mit exakt denselben Parametern wie Q-Learning
        eval_results = trainer.evaluate(episodes=config['episodes'], load_model=True)

        # Wiederherstellen der ursprünglichen max_steps
        if original_max_steps is not None:
            trainer.env.max_steps = original_max_steps

        # Konvertiere DQN-Ergebnisse zu Q-Learning-kompatibler Struktur
        results = convert_dqn_to_qlearning_format(eval_results, config['episodes'], config['max_steps'])

        print_scenario_result(scenario_name, results, config)
        return results

    except Exception as e:
        print(f"  ❌ Fehler bei DQN-Evaluation für {scenario_name}: {e}")
        return None


# ============================================================================
# Hauptfunktion - IDENTISCH zu Q-Learning
# ============================================================================

def compare_all_scenarios():
    """Hauptfunktion für DQN Szenarien-Vergleich"""
    # Setup
    config = load_evaluation_config()
    set_all_seeds()
    setup_export()

    print_config_header(config)
    print("\nStarte DQN-Evaluation aller Szenarien...")

    # Evaluation
    all_results = {}
    all_metrics = {}

    for scenario_name, scenario_config in SCENARIOS.items():
        results = evaluate_single_scenario(scenario_name, scenario_config, config)
        if results is not None:
            all_results[scenario_name] = results
            # Verwende Q-Learning calculate_metrics für Konsistenz
            from utils.evaluation import calculate_metrics
            all_metrics[scenario_name] = calculate_metrics(results, config['episodes'])
        else:
            print(f"  ⚠️  Überspringe {scenario_name} (DQN-Modell nicht verfügbar)")

    # Validierung
    if not all_metrics:
        print("\n❌ Keine DQN-Modelle für Vergleich gefunden!")
        print("Bitte führen Sie zuerst das Training für alle Szenarien aus:")
        print("  python src/dqn/train_all_scenarios.py")
        return None, None

    # Visualisierung
    print(f"\n{'=' * 60}")
    print("ERSTELLUNG DER VERGLEICHSGRAFIKEN")
    print(f"{'=' * 60}")

    # Verwende Q-Learning Visualisierungen OHNE zusätzliche Parameter
    from utils.visualization import (create_comparison_table, create_success_rate_comparison,
                                     create_stacked_failure_chart)

    create_comparison_table(all_metrics)
    create_success_rate_comparison(all_metrics)
    create_stacked_failure_chart(all_metrics)

    # DQN-spezifische wissenschaftliche Performance-Analyse
    create_scientific_performance_comparison(all_metrics, all_results, config)

    # Summary
    print(f"\n✅ DQN-Vergleich abgeschlossen!")
    print(f"Parameter: EVAL_MAX_STEPS={config['max_steps']}, LOOP_THRESHOLD={LOOP_THRESHOLD}")
    print(f"Erfolgreich evaluierte Szenarien: {list(all_metrics.keys())}")

    return all_results, all_metrics


def create_scientific_performance_comparison(all_metrics, all_results, config):
    """
    Erstellt wissenschaftliche 4-Panel Performance-Vergleich für DQN.
    IDENTISCH zur Q-Learning Version, nur Titel angepasst.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Wissenschaftliche Farbpalette (gedeckt, kontrastreich)
    colors = {
        'primary': '#2E86AB',  # Blau
        'secondary': 'lightcoral',  # Hellrosa
        'accent1': '#F18F01',  # Orange
        'accent2': '#C73E1D',  # Rot
        'neutral': '#5A5A5A'  # Grau
    }

    # Szenarien in wissenschaftlicher Reihenfolge
    scenarios = ['static', 'random_start', 'random_goal', 'random_obstacles', 'container']
    scenario_labels = ['Static', 'Random Start', 'Random Goal', 'Random Obstacles', 'Container']

    # Figure Setup - DQN-Titel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Deep Q-Learning Performance Vergleich aller Szenarien',
                 fontsize=16, fontweight='bold', y=0.95)

    # ===== PANEL 1: SUCCESS RATE =====
    ax1 = axes[0, 0]
    success_rates = []
    success_stds = []

    for scenario in scenarios:
        if scenario in all_metrics:
            # Berechne Success Rate aus rohen Daten für Genauigkeit
            raw_results = all_results[scenario]
            success_rate = (raw_results[RESULT_KEYS['SUCCESS_COUNT']] / config['episodes']) * 100

            # Für Std: Simuliere Episode-level Varianz (da wir nur einen Run haben)
            episode_successes = raw_results[RESULT_KEYS['SUCCESS_PER_EPISODE']]
            success_std = np.std(episode_successes) * 100  # In Prozent

            success_rates.append(success_rate)
            success_stds.append(success_std)
        else:
            success_rates.append(0)
            success_stds.append(0)

    x_pos = np.arange(len(scenarios))
    bars1 = ax1.bar(x_pos, success_rates, yerr=success_stds,
                    color=colors['primary'], alpha=0.8, capsize=4,
                    edgecolor='white', linewidth=1.2)

    ax1.set_title('A) Erfolgsrate (Success Rate)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Erfolgsrate (%)', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenario_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Werte auf Balken
    for i, (rate, std) in enumerate(zip(success_rates, success_stds)):
        if rate > 0:
            ax1.text(i, rate + std + 2, f'{rate:.1f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ===== PANEL 2: SAMPLE EFFICIENCY =====
    ax2 = axes[0, 1]
    avg_steps = []
    steps_stds = []

    for scenario in scenarios:
        if scenario in all_metrics:
            raw_results = all_results[scenario]
            steps_to_goal = raw_results[RESULT_KEYS['STEPS_TO_GOAL']]

            if steps_to_goal:
                avg_step = np.mean(steps_to_goal)
                std_step = np.std(steps_to_goal)
            else:
                avg_step = config['max_steps']  # Penalty für kein Erfolg
                std_step = 0

            avg_steps.append(avg_step)
            steps_stds.append(std_step)
        else:
            avg_steps.append(config['max_steps'])
            steps_stds.append(0)

    bars2 = ax2.bar(x_pos, avg_steps, yerr=steps_stds,
                    color=colors['secondary'], alpha=0.8, capsize=4,
                    edgecolor='white', linewidth=1.2)

    ax2.set_title('B) Durchschnittliche Anzahl Schritte (Sample Efficency)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Durchschn. Schritte zum Ziel', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenario_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Werte auf Balken
    for i, (steps, std) in enumerate(zip(avg_steps, steps_stds)):
        ax2.text(i, steps + std + 1, f'{steps:.1f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ===== PANEL 3: FAILURE ANALYSIS =====
    ax3 = axes[1, 0]

    # Failure kategorien sammeln
    failure_data = {
        'Timeout': [],
        'Loop Detection': [],
        'Obstacles': []
    }

    for scenario in scenarios:
        if scenario in all_results:
            raw_results = all_results[scenario]
            total_episodes = config['episodes']

            timeout_pct = (raw_results[RESULT_KEYS['TIMEOUT_COUNT']] / total_episodes) * 100
            loop_pct = (raw_results[RESULT_KEYS['LOOP_ABORT_COUNT']] / total_episodes) * 100
            obstacle_pct = (raw_results[RESULT_KEYS['OBSTACLE_COUNT']] / total_episodes) * 100

            failure_data['Timeout'].append(timeout_pct)
            failure_data['Loop Detection'].append(loop_pct)
            failure_data['Obstacles'].append(obstacle_pct)
        else:
            failure_data['Timeout'].append(0)
            failure_data['Loop Detection'].append(0)
            failure_data['Obstacles'].append(0)

    # Stacked Bar Chart
    bottom_timeout = np.array(failure_data['Timeout'])
    bottom_loop = bottom_timeout + np.array(failure_data['Loop Detection'])

    ax3.bar(x_pos, failure_data['Timeout'],
            color=colors['accent1'], alpha=0.8, label='Timeout',
            edgecolor='white', linewidth=1)
    ax3.bar(x_pos, failure_data['Loop Detection'], bottom=bottom_timeout,
            color=colors['accent2'], alpha=0.8, label='Loop Detection',
            edgecolor='white', linewidth=1)
    ax3.bar(x_pos, failure_data['Obstacles'], bottom=bottom_loop,
            color=colors['neutral'], alpha=0.8, label='Obstacles',
            edgecolor='white', linewidth=1)

    ax3.set_title('C) Fehlschlag-Rate (Failure Mode Analysis)', fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylabel('Fehlschlag-Rate (%)', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scenario_labels, rotation=45, ha='right')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ===== PANEL 4: PERFORMANCE STABILITY =====
    ax4 = axes[1, 1]

    # Simuliere Runs basierend auf Episode-Varianz (da nur ein Run verfügbar)
    stability_data = []

    for scenario in scenarios:
        if scenario in all_results:
            raw_results = all_results[scenario]
            episode_successes = raw_results[RESULT_KEYS['SUCCESS_PER_EPISODE']]

            # Simuliere 5 "Runs" durch Aufteilen der Episoden
            episodes_per_run = len(episode_successes) // 5
            run_success_rates = []

            for run in range(5):
                start_idx = run * episodes_per_run
                end_idx = start_idx + episodes_per_run
                run_episodes = episode_successes[start_idx:end_idx]

                if run_episodes:
                    run_success_rate = (sum(run_episodes) / len(run_episodes)) * 100
                else:
                    run_success_rate = 0

                run_success_rates.append(run_success_rate)

            stability_data.append(run_success_rates)
        else:
            stability_data.append([0, 0, 0, 0, 0])

    # Heatmap
    stability_matrix = np.array(stability_data)
    im = ax4.imshow(stability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax4.set_title('D) Leistungsstabilität (Performance Stability)', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xlabel('Evaluations-Durchlauf (Evaluation Run)', fontsize=12)
    ax4.set_yticks(range(len(scenarios)))
    ax4.set_yticklabels(scenario_labels)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels([f'Run {i + 1}' for i in range(5)])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Success Rate (%)', fontsize=11)

    # Werte in Heatmap
    for i in range(len(scenarios)):
        for j in range(5):
            text = ax4.text(j, i, f'{stability_matrix[i, j]:.0f}',
                            ha="center", va="center", color="black",
                            fontsize=9, fontweight='bold')

    # ===== LAYOUT FINALISIERUNG =====
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    # Speichern mit DQN-Prefix
    save_path = os.path.join("exports", "dqn_scientific_performance_analysis.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"DQN Wissenschaftliche Performance-Analyse gespeichert: {save_path}")
    plt.close()


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    compare_all_scenarios()