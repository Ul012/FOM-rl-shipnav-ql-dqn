# src/dqn/train_all_scenarios.py

import sys
import os
import time
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dqn.train import DQNTrainer
from src.shared.config import SETUP_NAME, EPISODES, EXPORT_PATH_DQN


class DQNScenarioRunner:
    """Führt DQN Training für alle Szenarien durch."""

    def __init__(self):
        self.scenarios = [
            'static',
            'random_start',
            'random_goal',
            'random_obstacles',
            'container'
        ]

        self.results = {}

    def run_all_scenarios(self, episodes: int = None, num_runs: int = 3) -> Dict[str, Any]:
        """
        Führt Training für alle Szenarien durch.

        Args:
            episodes: Anzahl Episoden pro Szenario
            num_runs: Anzahl Wiederholungen pro Szenario

        Returns:
            Dictionary mit allen Ergebnissen
        """
        if episodes is None:
            episodes = EPISODES

        print("=" * 60)
        print("DQN TRAINING FÜR ALLE SZENARIEN")
        print("=" * 60)
        print(f"Szenarien: {len(self.scenarios)}")
        print(f"Episoden pro Szenario: {episodes}")
        print(f"Runs pro Szenario: {num_runs}")
        print()

        all_results = {}
        summary_data = []

        for scenario in self.scenarios:
            print(f"\n{'=' * 50}")
            print(f"SZENARIO: {scenario.upper()}")
            print(f"{'=' * 50}")

            scenario_results = []

            for run in range(num_runs):
                print(f"\nRun {run + 1}/{num_runs}")
                print("-" * 30)

                start_time = time.time()

                # Trainer für dieses Szenario erstellen
                trainer = DQNTrainer(env_mode=scenario)

                # Training
                train_results = trainer.train(episodes=episodes)

                # Evaluation
                eval_results = trainer.evaluate(episodes=100, load_model=False)

                # Kombiniere Ergebnisse
                run_results = {
                    'run': run + 1,
                    'scenario': scenario,
                    'training': train_results,
                    'evaluation': eval_results,
                    'duration': time.time() - start_time
                }

                scenario_results.append(run_results)

                # Summary für diesen Run
                summary_data.append({
                    'scenario': scenario,
                    'run': run + 1,
                    'train_success_rate': train_results['success_rate'],
                    'eval_success_rate': eval_results['success_rate'],
                    'eval_avg_steps': eval_results['average_steps'],
                    'eval_avg_reward': eval_results['average_reward'],
                    'final_exploration': train_results['final_metrics']['exploration_rate'],
                    'memory_size': train_results['final_metrics']['memory_size'],
                    'avg_loss': train_results['final_metrics']['average_loss'],
                    'duration_minutes': run_results['duration'] / 60
                })

                print(f"Training Erfolgsrate: {train_results['success_rate']:.1f}%")
                print(f"Evaluation Erfolgsrate: {eval_results['success_rate']:.1f}%")
                print(f"Durchschnittliche Schritte: {eval_results['average_steps']:.1f}")
                print(f"Dauer: {run_results['duration'] / 60:.1f} Minuten")

                # Plot für diesen Run erstellen
                plot_path = os.path.join(EXPORT_PATH_DQN, SETUP_NAME, f'{SETUP_NAME}_dqn_training_{scenario}_run{run + 1}.pdf')
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                trainer.plot_training_results(train_results, plot_path)

            all_results[scenario] = scenario_results

            # Szenario-Zusammenfassung
            self._print_scenario_summary(scenario, scenario_results)

        # Gesamtauswertung
        self._print_overall_summary(summary_data)

        # Speichere detaillierte Ergebnisse
        self._save_results(all_results, summary_data)

        # Erstelle kombinierte Kurven - IDENTISCH zu Q-Learning
        self._create_combined_curve_plots(all_results)

        return {
            'detailed_results': all_results,
            'summary': summary_data
        }

    def _print_scenario_summary(self, scenario: str, results: List[Dict]):
        """Gibt Zusammenfassung für ein Szenario aus."""
        print(f"\n{'=' * 30}")
        print(f"ZUSAMMENFASSUNG {scenario.upper()}")
        print(f"{'=' * 30}")

        # Berechne Durchschnitte
        train_success_rates = [r['training']['success_rate'] for r in results]
        eval_success_rates = [r['evaluation']['success_rate'] for r in results]
        eval_steps = [r['evaluation']['average_steps'] for r in results]
        eval_rewards = [r['evaluation']['average_reward'] for r in results]
        durations = [r['duration'] for r in results]

        print(f"Training Erfolgsrate: {sum(train_success_rates) / len(train_success_rates):.1f}% "
              f"(± {pd.Series(train_success_rates).std():.1f}%)")
        print(f"Evaluation Erfolgsrate: {sum(eval_success_rates) / len(eval_success_rates):.1f}% "
              f"(± {pd.Series(eval_success_rates).std():.1f}%)")
        print(f"Durchschnittliche Schritte: {sum(eval_steps) / len(eval_steps):.1f} "
              f"(± {pd.Series(eval_steps).std():.1f})")
        print(f"Durchschnittliche Belohnung: {sum(eval_rewards) / len(eval_rewards):.2f} "
              f"(± {pd.Series(eval_rewards).std():.2f})")
        print(f"Durchschnittliche Dauer: {sum(durations) / len(durations) / 60:.1f} Minuten")

    def _print_overall_summary(self, summary_data: List[Dict]):
        """Gibt Gesamtübersicht aus."""
        print(f"\n{'=' * 80}")
        print("GESAMTAUSWERTUNG ALLER SZENARIEN")
        print(f"{'=' * 80}")

        df = pd.DataFrame(summary_data)

        # Gruppiere nach Szenarien
        grouped = df.groupby('scenario').agg({
            'eval_success_rate': ['mean', 'std'],
            'eval_avg_steps': ['mean', 'std'],
            'eval_avg_reward': ['mean', 'std'],
            'duration_minutes': ['mean', 'std']
        }).round(2)

        print("\nErfolgsrate (Evaluation):")
        print(f"{'Szenario':<15} {'Mittelwert':<12} {'Std.Abw.':<10}")
        print("-" * 40)
        for scenario in self.scenarios:
            if scenario in grouped.index:
                mean_val = grouped.loc[scenario, ('eval_success_rate', 'mean')]
                std_val = grouped.loc[scenario, ('eval_success_rate', 'std')]
                print(f"{scenario:<15} {mean_val:>8.1f}% {std_val:>8.1f}%")

        print("\nDurchschnittliche Schritte:")
        print(f"{'Szenario':<15} {'Mittelwert':<12} {'Std.Abw.':<10}")
        print("-" * 40)
        for scenario in self.scenarios:
            if scenario in grouped.index:
                mean_val = grouped.loc[scenario, ('eval_avg_steps', 'mean')]
                std_val = grouped.loc[scenario, ('eval_avg_steps', 'std')]
                print(f"{scenario:<15} {mean_val:>8.1f} {std_val:>8.1f}")

        print("\nDurchschnittliche Belohnung:")
        print(f"{'Szenario':<15} {'Mittelwert':<12} {'Std.Abw.':<10}")
        print("-" * 40)
        for scenario in self.scenarios:
            if scenario in grouped.index:
                mean_val = grouped.loc[scenario, ('eval_avg_reward', 'mean')]
                std_val = grouped.loc[scenario, ('eval_avg_reward', 'std')]
                print(f"{scenario:<15} {mean_val:>8.2f} {std_val:>8.2f}")

    def _save_results(self, all_results: Dict, summary_data: List[Dict]):
        """Speichert Ergebnisse in Dateien."""
        # CSV mit Zusammenfassung (im SETUP_NAME Unterordner)
        export_dir = os.path.join(EXPORT_PATH_DQN, SETUP_NAME)
        os.makedirs(export_dir, exist_ok=True)

        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(export_dir, 'dqn_all_scenarios_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nZusammenfassung gespeichert: {csv_path}")

        # Detaillierte Statistiken
        stats_path = os.path.join(export_dir, 'dqn_all_scenarios_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("DQN ALL SCENARIOS STATISTICS\n")
            f.write("=" * 50 + "\n\n")

            for scenario, results in all_results.items():
                f.write(f"SCENARIO: {scenario.upper()}\n")
                f.write("-" * 30 + "\n")

                for i, result in enumerate(results):
                    f.write(f"Run {i + 1}:\n")
                    f.write(f"  Training Success Rate: {result['training']['success_rate']:.1f}%\n")
                    f.write(f"  Evaluation Success Rate: {result['evaluation']['success_rate']:.1f}%\n")
                    f.write(f"  Average Steps: {result['evaluation']['average_steps']:.1f}\n")
                    f.write(f"  Average Reward: {result['evaluation']['average_reward']:.2f}\n")
                    f.write(f"  Duration: {result['duration'] / 60:.1f} minutes\n\n")

                f.write("\n")

        print(f"Detaillierte Statistiken gespeichert: {stats_path}")

    def _create_combined_curve_plots(self, all_results: Dict):
        """
        Erstellt kombinierte Learning- und Success-Kurven für alle Szenarien.
        IDENTISCH zur Q-Learning Version.
        """
        # Kombinierte Learning Curve (Rewards)
        self._create_combined_learning_curve(all_results)

        # Kombinierte Success Curve
        self._create_combined_success_curve(all_results)

    def _create_combined_learning_curve(self, all_results: Dict):
        """Erstellt kombinierte Learning Curve für alle Szenarien - IDENTISCH zu Q-Learning"""

        plt.figure(figsize=(10, 6))

        # Farben für Szenarien - IDENTISCHE Reihenfolge und Farben wie Q-Learning
        scenario_names = ['static', 'random_start', 'random_goal', 'random_obstacles', 'container']

        for i, scenario in enumerate(scenario_names):
            if scenario in all_results:
                # Nimm den ersten Run für Konsistenz (oder Durchschnitt über alle Runs)
                scenario_results = all_results[scenario]
                if scenario_results:
                    # Verwende ersten Run
                    first_run = scenario_results[0]
                    episode_rewards = first_run['training']['episode_rewards']

                    # Gleitender Durchschnitt für Glättung - IDENTISCH zu Q-Learning
                    window_size = 50
                    if len(episode_rewards) >= window_size:
                        smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size) / window_size,
                                                       mode='valid')
                    else:
                        smoothed_rewards = episode_rewards

                    label = scenario.replace("_", " ").capitalize()
                    plt.plot(smoothed_rewards, label=label, linewidth=1.2, alpha=0.8)

        plt.title('Learning Curve (DQN)', fontweight='bold')
        plt.xlabel("Episode", fontweight='bold')
        plt.ylabel("Reward", fontweight='bold')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # plt.tight_layout()

        # Speichern im combined Unterordner - IDENTISCH zu Q-Learning
        combined_dir = os.path.join(EXPORT_PATH_DQN, SETUP_NAME, 'combined')
        os.makedirs(combined_dir, exist_ok=True)

        save_path = os.path.join(combined_dir, f'{SETUP_NAME}_scenario-comparison_learning_curve.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kombinierte Learning Curve gespeichert: {save_path}")
        plt.close()

    def _create_combined_success_curve(self, all_results: Dict):
        """Erstellt kombinierte Success Curve für alle Szenarien - IDENTISCH zu Q-Learning"""
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(10, 6))

        # Farben für Szenarien - IDENTISCHE Reihenfolge und Farben wie Q-Learning
        scenario_names = ['static', 'random_start', 'random_goal', 'random_obstacles', 'container']

        for i, scenario in enumerate(scenario_names):
            if scenario in all_results:
                scenario_results = all_results[scenario]
                if scenario_results:
                    # Verwende ersten Run
                    first_run = scenario_results[0]
                    episode_successes = first_run['training']['episode_successes']

                    # Konvertiere Boolean zu Integer für Berechnung
                    success_values = [1 if success else 0 for success in episode_successes]

                    # Gleitender Durchschnitt für Success Rate
                    window_size = 50
                    if len(success_values) >= window_size:
                        smoothed_success = np.convolve(success_values, np.ones(window_size) / window_size, mode='valid')
                    else:
                        smoothed_success = success_values

                    label = scenario.replace("_", " ").capitalize()
                    plt.plot(smoothed_success, label=label, linewidth=1.2, alpha=0.8)

        plt.title('Success Rate Curve (DQN)', fontweight='bold')
        plt.xlabel("Episode", fontweight='bold')
        plt.ylabel("Success Rate", fontweight='bold')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # plt.tight_layout()

        # Speichern im combined Unterordner - IDENTISCH zu Q-Learning
        combined_dir = os.path.join(EXPORT_PATH_DQN, SETUP_NAME, 'combined')
        os.makedirs(combined_dir, exist_ok=True)

        save_path = os.path.join(combined_dir, f'{SETUP_NAME}_scenario-comparison_success_curve.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kombinierte Success Curve gespeichert: {save_path}")
        plt.close()


def main():
    """Hauptfunktion für das Training aller Szenarien."""
    import argparse

    parser = argparse.ArgumentParser(description='DQN Training für alle Szenarien')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Anzahl Episoden pro Szenario')
    parser.add_argument('--runs', type=int, default=3,
                        help='Anzahl Wiederholungen pro Szenario')
    parser.add_argument('--scenarios', nargs='+',
                        choices=['static', 'random_start', 'random_goal', 'random_obstacles', 'container'],
                        help='Spezifische Szenarien (default: alle)')

    args = parser.parse_args()

    # Runner erstellen
    runner = DQNScenarioRunner()

    # Szenarien filtern falls spezifiziert
    if args.scenarios:
        runner.scenarios = args.scenarios
        print(f"Ausgewählte Szenarien: {args.scenarios}")

    # Training starten
    start_time = time.time()
    results = runner.run_all_scenarios(episodes=args.episodes, num_runs=args.runs)
    total_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("ALLE SZENARIEN ABGESCHLOSSEN")
    print(f"{'=' * 60}")
    print(f"Gesamtdauer: {total_time / 3600:.1f} Stunden")
    print(f"Durchschnitt pro Szenario: {total_time / (len(runner.scenarios) * args.runs) / 60:.1f} Minuten")

    return results


if __name__ == "__main__":
    main()