# utils/reporting.py

import numpy as np
from src.shared.config import EPISODES, ENV_MODE, ALPHA, GAMMA, EPSILON, SEED, EXPORT_PDF, EXPORT_PATH


# Ausgabe der Trainingsergebnisse
def print_training_results(rewards_per_episode, success_per_episode, steps_per_episode):
    total_successes = sum(success_per_episode)
    avg_reward = np.mean(rewards_per_episode)
    avg_steps = np.mean(steps_per_episode)

    print(f"\n" + "=" * 60)
    print(f"TRAININGSERGEBNISSE ({EPISODES} Episoden, Modus: {ENV_MODE})")
    print("=" * 60)

    print(f"\nErfolgsstatistik:")
    print(f"  Erfolgreiche Episoden: {total_successes}/{EPISODES} ({(total_successes / EPISODES) * 100:.1f}%)")

    # Erfolgsrate in verschiedenen Phasen
    phase_size = min(500, EPISODES // 4)
    if len(success_per_episode) >= phase_size * 2:
        early_success = np.mean(success_per_episode[:phase_size]) * 100
        late_success = np.mean(success_per_episode[-phase_size:]) * 100
        print(f"  Frühe Phase (erste {phase_size}): {early_success:.1f}%")
        print(f"  Späte Phase (letzte {phase_size}): {late_success:.1f}%")
        print(f"  Verbesserung: {late_success - early_success:+.1f} Prozentpunkte")

    print(f"\nReward-Statistiken:")
    print(f"  Durchschnitt: {avg_reward:.2f}")
    print(f"  Minimum: {min(rewards_per_episode):.2f}")
    print(f"  Maximum: {max(rewards_per_episode):.2f}")
    print(f"  Standardabweichung: {np.std(rewards_per_episode):.2f}")

    print(f"\nSchritt-Statistiken:")
    print(f"  Durchschnittliche Schritte: {avg_steps:.1f}")
    print(f"  Minimum: {min(steps_per_episode)}")
    print(f"  Maximum: {max(steps_per_episode)}")

    print(f"\nHyperparameter:")
    print(f"  Lernrate (α): {ALPHA}")
    print(f"  Discount Factor (γ): {GAMMA}")
    print(f"  Epsilon (ε): {EPSILON}")
    print(f"  Seed: {SEED}")

    if EXPORT_PDF:
        print(f"\nPDF-Exports gespeichert in: {EXPORT_PATH}")


# Ausgabe der Evaluationsergebnisse
def print_evaluation_results(results_cause, results_solved, rewards_all, episodes, env_mode):
    avg_reward = np.mean(rewards_all)

    print(f"\n" + "=" * 60)
    print(f"EVALUATIONSERGEBNISSE ({episodes} Episoden, Modus: {env_mode})")
    print("=" * 60)

    print(f"\nDetaillierte Ursachen-Verteilung:")
    total_episodes = sum(results_cause.values())
    for cause, count in sorted(results_cause.items()):
        percentage = (count / total_episodes) * 100
        print(f"  {cause}: {count} ({percentage:.1f}%)")

    print(f"\nLösungsrate:")
    solved_count = results_solved["solved episode"]
    failed_count = results_solved["failed episode"]
    print(f"  Erfolgreich: {solved_count}/{episodes} ({(solved_count / episodes) * 100:.1f}%)")
    print(f"  Fehlgeschlagen: {failed_count}/{episodes} ({(failed_count / episodes) * 100:.1f}%)")

    print(f"\nReward-Statistiken:")
    print(f"  Durchschnitt: {avg_reward:.2f}")
    print(f"  Minimum: {min(rewards_all):.2f}")
    print(f"  Maximum: {max(rewards_all):.2f}")
    print(f"  Median: {np.median(rewards_all):.2f}")

    if EXPORT_PDF:
        print(f"\nPDF-Exports gespeichert in: {EXPORT_PATH}")