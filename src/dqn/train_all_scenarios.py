# src/dqn/train_all_scenarios.py

import sys
import os
import time
import pandas as pd
from typing import Dict, List, Any

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dqn.train import DQNTrainer
from src.shared.config import DQN_EPISODES, EXPORT_PATH


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
            episodes = DQN_EPISODES

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
                plot_path = os.path.join(EXPORT_PATH, f'dqn_training_{scenario}_run{run + 1}.pdf')
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                trainer.plot_training_results(train_results, plot_path)

            all_results[scenario] = scenario_results

            # Szenario-Zusammenfassung
            self._print_scenario_summary(scenario, scenario_results)

        # Gesamtauswertung
        self._print_overall_summary(summary_data)

        # Speichere detaillierte Ergebnisse
        self._save_results(all_results, summary_data)

        # Erstelle Vergleichs-Visualisierung
        self._create_comparison_plots(summary_data)

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
        # CSV mit Zusammenfassung
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(EXPORT_PATH, 'dqn_all_scenarios_summary.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nZusammenfassung gespeichert: {csv_path}")

        # Detaillierte Statistiken
        stats_path = os.path.join(EXPORT_PATH, 'dqn_all_scenarios_stats.txt')
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

    def _create_comparison_plots(self, summary_data: List[Dict]):
        """Erstellt Vergleichs-Visualisierungen."""
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.DataFrame(summary_data)

        # Gruppiere nach Szenarien
        grouped = df.groupby('scenario').agg({
            'eval_success_rate': ['mean', 'std'],
            'eval_avg_steps': ['mean', 'std'],
            'eval_avg_reward': ['mean', 'std']
        })

        # Plot erstellen
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Performance Vergleich aller Szenarien', fontsize=16)

        scenarios = list(grouped.index)
        x = np.arange(len(scenarios))

        # Success Rate
        means = [grouped.loc[s, ('eval_success_rate', 'mean')] for s in scenarios]
        stds = [grouped.loc[s, ('eval_success_rate', 'std')] for s in scenarios]

        axes[0, 0].bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Erfolgsrate')
        axes[0, 0].set_ylabel('Erfolgsrate (%)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scenarios, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Average Steps
        means = [grouped.loc[s, ('eval_avg_steps', 'mean')] for s in scenarios]
        stds = [grouped.loc[s, ('eval_avg_steps', 'std')] for s in scenarios]

        axes[0, 1].bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Durchschnittliche Schritte')
        axes[0, 1].set_ylabel('Schritte')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scenarios, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Average Reward
        means = [grouped.loc[s, ('eval_avg_reward', 'mean')] for s in scenarios]
        stds = [grouped.loc[s, ('eval_avg_reward', 'std')] for s in scenarios]

        axes[1, 0].bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Durchschnittliche Belohnung')
        axes[1, 0].set_ylabel('Belohnung')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(scenarios, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Heatmap: Success Rate pro Run
        heatmap_data = []
        for scenario in scenarios:
            scenario_data = df[df['scenario'] == scenario]['eval_success_rate'].values
            heatmap_data.append(scenario_data)

        # Pad arrays to same length
        max_runs = max(len(row) for row in heatmap_data)
        padded_data = []
        for row in heatmap_data:
            if len(row) < max_runs:
                padded_row = np.pad(row, (0, max_runs - len(row)), constant_values=np.nan)
            else:
                padded_row = row
            padded_data.append(padded_row)

        im = axes[1, 1].imshow(padded_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        axes[1, 1].set_title('Erfolgsrate pro Run')
        axes[1, 1].set_yticks(range(len(scenarios)))
        axes[1, 1].set_yticklabels(scenarios)
        axes[1, 1].set_xlabel('Run')

        # Colorbar
        plt.colorbar(im, ax=axes[1, 1], label='Erfolgsrate (%)')

        plt.tight_layout()

        # Speichern
        plot_path = os.path.join(EXPORT_PATH, 'dqn_all_scenarios_comparison.pdf')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Vergleichs-Plot gespeichert: {plot_path}")
#       plt.show() # alle Visualisierungen werden interaktiv angezeigt
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