# src/comparison/legacy_algorithm_comparison_visualization.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# Importiere die Module erst in den Methoden, wenn sie gebraucht werden
# Das vermeidet Import-Probleme beim Laden des Scripts


class AlgorithmComparator:
    """Vergleicht Q-Learning und DQN Algorithmen über mehrere Szenarien."""

    def __init__(self):
        self.scenarios = ['static', 'random_start', 'random_goal', 'random_obstacles', 'container']
        self.scenario_labels = ['Static', 'Random Start', 'Random Goal', 'Random Obstacles', 'Container']
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Lädt Evaluationskonfiguration."""
        from src.shared.config_utils import get_shared_config
        shared_config = get_shared_config()
        return {
            'episodes': shared_config['eval_episodes'],
            'max_steps': shared_config['eval_max_steps'],
            'seed': shared_config['seed']
        }

    def run_comparison(self) -> None:
        """Führt kompletten Algorithmusvergleich durch."""
        print("🚀 ALGORITHMUS-VERGLEICHSANALYSE")
        print("=" * 50)

        # Evaluierungen durchführen
        ql_results, ql_metrics = self._run_qlearning_evaluation()
        dqn_results, dqn_metrics = self._run_dqn_evaluation()

        # Validierung
        if not ql_results or not dqn_results:
            self._print_missing_data_warning(ql_results, dqn_results)
            return

        # Vergleichsvisualisierung erstellen
        self._create_visualization(ql_results, ql_metrics, dqn_results, dqn_metrics)
        self._print_summary(ql_results, dqn_results)

    def _run_qlearning_evaluation(self) -> Tuple[Dict, Dict]:
        """Führt Q-Learning Evaluierung aus - direkte Implementierung wie in compare_algorithms_v1.py."""
        print("\n📊 Q-Learning Evaluierung...")

        try:
            # Importiere nur die benötigten Utilities - KEINE Visualisierungsfunktionen
            import sys
            import os
            import numpy as np

            # Füge Q-Learning utils Pfad hinzu
            ql_utils_path = os.path.join(os.path.dirname(__file__), "../q_learning")
            sys.path.insert(0, ql_utils_path)

            from utils.common import set_all_seeds, obs_to_state, check_success
            from utils.environment import initialize_environment_for_scenario
            from utils.qlearning import load_q_table
            from utils.evaluation import check_loop_detection
            from src.shared.config_utils import get_q_table_path, get_shared_config

            # Konfiguration laden
            shared_config = get_shared_config()
            config = {
                'episodes': shared_config['eval_episodes'],
                'max_steps': shared_config['eval_max_steps'],
                'seed': shared_config['seed']
            }

            # Szenarien definieren
            scenarios = {
                "static": {"env_mode": "static", "environment": "grid"},
                "random_start": {"env_mode": "random_start", "environment": "grid"},
                "random_goal": {"env_mode": "random_goal", "environment": "grid"},
                "random_obstacles": {"env_mode": "random_obstacles", "environment": "grid"},
                "container": {"env_mode": "container", "environment": "container"}
            }

            ql_results = {}
            ql_metrics = {}

            # Seeds setzen
            set_all_seeds()

            print(f"   Evaluiere {len(scenarios)} Q-Learning Szenarien...")

            for scenario_name, scenario_config in scenarios.items():
                try:
                    # Prüfe ob Q-Table existiert
                    q_table_path = get_q_table_path(scenario_config["env_mode"])
                    full_path = os.path.join(ql_utils_path, q_table_path)

                    if not os.path.exists(full_path):
                        print(f"     ✗ {scenario_name}: Q-Table nicht gefunden")
                        continue

                    # Evaluierung durchführen - DIREKT implementiert
                    results = self._evaluate_qlearning_scenario(
                        scenario_name, scenario_config, config,
                        initialize_environment_for_scenario, load_q_table,
                        obs_to_state, check_success, check_loop_detection,
                        q_table_path, ql_utils_path
                    )

                    if results:
                        ql_results[scenario_name] = results
                        success_rate = (results['success_count'] / config['episodes']) * 100
                        ql_metrics[scenario_name] = {'success_rate': success_rate}
                        print(f"     ✓ {scenario_name}: {success_rate:.1f}% Erfolgsrate")

                except Exception as e:
                    print(f"     ✗ {scenario_name}: Fehler - {e}")
                    continue

            # Cleanup
            sys.path.remove(ql_utils_path)

            print(f"   Q-Learning: {len(ql_results)} Szenarien erfolgreich evaluiert")
            return ql_results, ql_metrics

        except Exception as e:
            print(f"❌ Q-Learning Fehler: {e}")
            return {}, {}

    def _evaluate_qlearning_scenario(self, scenario_name, scenario_config, config,
                                     initialize_environment_for_scenario, load_q_table,
                                     obs_to_state, check_success, check_loop_detection,
                                     q_table_path, ql_utils_path):
        """Evaluiert ein einzelnes Q-Learning Szenario - direkte Implementierung."""
        import numpy as np

        # Environment und Q-Table laden
        env, grid_size = initialize_environment_for_scenario(scenario_config)
        Q = load_q_table(os.path.join(ql_utils_path, q_table_path))

        if Q is None:
            return None

        # Ergebnisse sammeln
        results = {
            'success_count': 0,
            'timeout_count': 0,
            'loop_abort_count': 0,
            'obstacle_count': 0,
            'episode_rewards': [],
            'steps_to_goal': [],
            'success_per_episode': []
        }

        # Episoden durchführen
        for episode in range(config['episodes']):
            obs, _ = env.reset()
            state = obs_to_state(obs, scenario_config["env_mode"], grid_size)
            episode_reward = 0
            steps = 0
            visited_states = {}

            for step in range(config['max_steps']):
                action = np.argmax(Q[state])
                obs, reward, terminated, _, _ = env.step(action)
                next_state = obs_to_state(obs, scenario_config["env_mode"], grid_size)
                episode_reward += reward
                steps += 1

                if check_success(reward, scenario_config["env_mode"]):
                    results['success_count'] += 1
                    results['steps_to_goal'].append(steps)
                    results['success_per_episode'].append(1)
                    break

                if check_loop_detection(visited_states, next_state, scenario_config["env_mode"]):
                    results['loop_abort_count'] += 1
                    results['success_per_episode'].append(0)
                    break

                if terminated:
                    results['obstacle_count'] += 1
                    results['success_per_episode'].append(0)
                    break

                state = next_state
            else:
                results['timeout_count'] += 1
                results['success_per_episode'].append(0)

            results['episode_rewards'].append(episode_reward)

        return results

    def _run_dqn_evaluation(self) -> Tuple[Dict, Dict]:
        """Führt DQN Evaluierung aus - direkte Implementierung wie in compare_algorithms_v1.py."""
        print("\n🧠 DQN Evaluierung...")

        try:
            # Direkte DQN Evaluierung ohne Visualisierungen
            from src.dqn.train import DQNTrainer
            from src.shared.config_utils import get_dqn_model_path, get_shared_config
            import numpy as np
            import os

            # Konfiguration laden
            shared_config = get_shared_config()
            config = {
                'episodes': shared_config['eval_episodes'],
                'max_steps': shared_config['eval_max_steps'],
                'seed': shared_config['seed']
            }

            # Szenarien definieren
            scenarios = {
                "static": {"env_mode": "static", "environment": "grid"},
                "random_start": {"env_mode": "random_start", "environment": "grid"},
                "random_goal": {"env_mode": "random_goal", "environment": "grid"},
                "random_obstacles": {"env_mode": "random_obstacles", "environment": "grid"},
                "container": {"env_mode": "container", "environment": "container"}
            }

            dqn_results = {}
            dqn_metrics = {}

            print(f"   Evaluiere {len(scenarios)} DQN Szenarien...")

            for scenario_name, scenario_config in scenarios.items():
                try:
                    # Prüfe ob DQN-Modell existiert
                    model_path = get_dqn_model_path(scenario_config["env_mode"])
                    full_path = os.path.join(os.path.dirname(__file__), "../dqn", model_path)

                    if not os.path.exists(full_path):
                        print(f"     ✗ {scenario_name}: DQN-Modell nicht gefunden")
                        continue

                    # DQN Trainer erstellen und evaluieren
                    trainer = DQNTrainer(env_mode=scenario_config["env_mode"])

                    # Nur Evaluierung, keine Visualisierung
                    eval_results = trainer.evaluate(episodes=config['episodes'], load_model=True)

                    # Konvertiere zu unserem Format
                    results = self._convert_dqn_results(eval_results, config)

                    if results:
                        dqn_results[scenario_name] = results
                        success_rate = (results['success_count'] / config['episodes']) * 100
                        dqn_metrics[scenario_name] = {'success_rate': success_rate}
                        print(f"     ✓ {scenario_name}: {success_rate:.1f}% Erfolgsrate")

                except Exception as e:
                    print(f"     ✗ {scenario_name}: Fehler - {e}")
                    continue

            print(f"   DQN: {len(dqn_results)} Szenarien erfolgreich evaluiert")
            return dqn_results, dqn_metrics

        except Exception as e:
            print(f"❌ DQN Fehler: {e}")
            return {}, {}

    def _convert_dqn_results(self, eval_results, config):
        """Konvertiert DQN Ergebnisse in unser Standard-Format."""
        if not eval_results:
            return None

        # Berechne success_count aus success_rate
        success_rate = eval_results.get('success_rate', 0)
        success_count = int((success_rate / 100) * config['episodes'])

        # Simuliere andere Zählungen (DQN unterscheidet nicht zwischen timeout/loop/obstacle)
        failure_count = config['episodes'] - success_count

        # Erstelle Dummy steps_to_goal basierend auf average_steps
        avg_steps = eval_results.get('average_steps', config['max_steps'])
        steps_to_goal = [avg_steps] * success_count if success_count > 0 else []

        # Erstelle success_per_episode Array
        success_per_episode = [1] * success_count + [0] * failure_count

        return {
            'success_count': success_count,
            'timeout_count': failure_count,  # DQN klassifiziert alle Failures als timeout
            'loop_abort_count': 0,
            'obstacle_count': 0,
            'episode_rewards': [eval_results.get('average_reward', 0)] * config['episodes'],
            'steps_to_goal': steps_to_goal,
            'success_per_episode': success_per_episode
        }

    def _print_missing_data_warning(self, ql_results: Dict, dqn_results: Dict) -> None:
        """Gibt Warnung bei fehlenden Daten aus."""
        if not ql_results:
            print("⚠️  Q-Learning Training erforderlich: python src/q_learning/train_all_scenarios.py")
        if not dqn_results:
            print("⚠️  DQN Training erforderlich: python src/dqn/train_all_scenarios.py")

    def _create_visualization(self, ql_results: Dict, ql_metrics: Dict,
                              dqn_results: Dict, dqn_metrics: Dict) -> None:
        """Erstellt wissenschaftliche Vergleichsvisualisierung."""
        print("\n📈 Erstelle Vergleichsvisualisierung...")

        visualizer = ComparisonVisualizer(self.scenarios, self.scenario_labels, self.config)
        visualizer.create_comparison_plot(ql_results, ql_metrics, dqn_results, dqn_metrics)

        print("✅ Visualisierung erstellt: exports/algorithm_comparison_analysis.pdf")

    def _print_summary(self, ql_results: Dict, dqn_results: Dict) -> None:
        """Gibt Ergebniszusammenfassung aus."""
        common_scenarios = set(ql_results.keys()) & set(dqn_results.keys())

        print(f"\n📊 ERGEBNISSE ({len(common_scenarios)} Szenarien):")
        print(f"{'Szenario':<16} {'Q-Learning':<10} {'DQN':<10} {'Gewinner':<12}")
        print("-" * 50)

        for scenario in sorted(common_scenarios):
            ql_rate = (ql_results[scenario]['success_count'] / self.config['episodes']) * 100
            dqn_rate = (dqn_results[scenario]['success_count'] / self.config['episodes']) * 100

            if ql_rate > dqn_rate:
                winner = "Q-Learning"
            elif dqn_rate > ql_rate:
                winner = "DQN"
            else:
                winner = "Tie"

            print(f"{scenario:<16} {ql_rate:>6.1f}% {dqn_rate:>6.1f}% {winner:<12}")


class ComparisonVisualizer:
    """Erstellt wissenschaftliche Vergleichsvisualisierungen."""

    def __init__(self, scenarios: list, scenario_labels: list, config: Dict):
        self.scenarios = scenarios
        self.scenario_labels = scenario_labels
        self.config = config
        self.colors = {
            'qlearning': '#1f77b4',
            'dqn': '#ff7f0e',
            'neutral': '#7f7f7f',
            'positive': '#2ca02c',
            'negative': '#d62728'
        }

    def create_comparison_plot(self, ql_results: Dict, ql_metrics: Dict,
                               dqn_results: Dict, dqn_metrics: Dict) -> None:
        """Erstellt 6-Panel Vergleichsplot."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Q-Learning vs Deep Q-Learning: Performance Analysis',
                     fontsize=16, fontweight='bold', y=0.95)

        # Daten vorbereiten
        ql_success, dqn_success = self._calculate_success_rates(ql_results, dqn_results)
        ql_steps, dqn_steps = self._calculate_step_efficiency(ql_results, dqn_results)

        # 6 Panels erstellen
        self._create_success_rate_panel(axes[0, 0], ql_success, dqn_success)
        self._create_efficiency_panel(axes[0, 1], ql_steps, dqn_steps)
        self._create_robustness_panel(axes[0, 2], ql_results, dqn_results)
        self._create_difficulty_panel(axes[1, 0], ql_success, dqn_success)
        self._create_stability_panel(axes[1, 1], ql_results, dqn_results)
        self._create_superiority_panel(axes[1, 2], ql_success, dqn_success)

        # Layout und Speichern
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        self._save_plot()
        plt.close()

    def _calculate_success_rates(self, ql_results: Dict, dqn_results: Dict) -> Tuple[list, list]:
        """Berechnet Erfolgsraten für beide Algorithmen."""
        ql_success = []
        dqn_success = []

        for scenario in self.scenarios:
            ql_rate = (ql_results[scenario]['success_count'] / self.config[
                'episodes']) * 100 if scenario in ql_results else 0
            dqn_rate = (dqn_results[scenario]['success_count'] / self.config[
                'episodes']) * 100 if scenario in dqn_results else 0

            ql_success.append(ql_rate)
            dqn_success.append(dqn_rate)

        return ql_success, dqn_success

    def _calculate_step_efficiency(self, ql_results: Dict, dqn_results: Dict) -> Tuple[list, list]:
        """Berechnet durchschnittliche Schritte zum Ziel."""
        ql_steps = []
        dqn_steps = []

        for scenario in self.scenarios:
            # Q-Learning
            if scenario in ql_results and ql_results[scenario]['steps_to_goal']:
                ql_avg = np.mean(ql_results[scenario]['steps_to_goal'])
            else:
                ql_avg = self.config['max_steps']
            ql_steps.append(ql_avg)

            # DQN
            if scenario in dqn_results and dqn_results[scenario]['steps_to_goal']:
                dqn_avg = np.mean(dqn_results[scenario]['steps_to_goal'])
            else:
                dqn_avg = self.config['max_steps']
            dqn_steps.append(dqn_avg)

        return ql_steps, dqn_steps

    def _create_success_rate_panel(self, ax, ql_success: list, dqn_success: list) -> None:
        """Panel A: Erfolgsratenvergleich."""
        x_pos = np.arange(len(self.scenarios))
        width = 0.35

        ax.bar(x_pos - width / 2, ql_success, width, label='Q-Learning',
               color=self.colors['qlearning'], alpha=0.8)
        ax.bar(x_pos + width / 2, dqn_success, width, label='DQN',
               color=self.colors['dqn'], alpha=0.8)

        ax.set_title('A) Erfolgsrate Vergleich', fontweight='bold')
        ax.set_ylabel('Erfolgsrate (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.scenario_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        # Werte anzeigen
        for i, (ql_val, dqn_val) in enumerate(zip(ql_success, dqn_success)):
            if ql_val > 0:
                ax.text(i - width / 2, ql_val + 2, f'{ql_val:.0f}%', ha='center', va='bottom', fontsize=9)
            if dqn_val > 0:
                ax.text(i + width / 2, dqn_val + 2, f'{dqn_val:.0f}%', ha='center', va='bottom', fontsize=9)

    def _create_efficiency_panel(self, ax, ql_steps: list, dqn_steps: list) -> None:
        """Panel B: Effizienzvergleich."""
        x_pos = np.arange(len(self.scenarios))
        width = 0.35

        ax.bar(x_pos - width / 2, ql_steps, width, label='Q-Learning',
               color=self.colors['qlearning'], alpha=0.8)
        ax.bar(x_pos + width / 2, dqn_steps, width, label='DQN',
               color=self.colors['dqn'], alpha=0.8)

        ax.set_title('B) Sample Effizienz', fontweight='bold')
        ax.set_ylabel('Durchschn. Schritte zum Ziel')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.scenario_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _create_robustness_panel(self, ax, ql_results: Dict, dqn_results: Dict) -> None:
        """Panel C: Robustheitsvergleich (Varianz)."""
        x_pos = np.arange(len(self.scenarios))
        width = 0.35

        ql_variance = []
        dqn_variance = []

        for scenario in self.scenarios:
            # Q-Learning Varianz
            if scenario in ql_results:
                ql_var = np.std(ql_results[scenario]['success_per_episode']) * 100
            else:
                ql_var = 0
            ql_variance.append(ql_var)

            # DQN Varianz
            if scenario in dqn_results:
                dqn_var = np.std(dqn_results[scenario]['success_per_episode']) * 100
            else:
                dqn_var = 0
            dqn_variance.append(dqn_var)

        ax.bar(x_pos - width / 2, ql_variance, width, label='Q-Learning',
               color=self.colors['qlearning'], alpha=0.8)
        ax.bar(x_pos + width / 2, dqn_variance, width, label='DQN',
               color=self.colors['dqn'], alpha=0.8)

        ax.set_title('C) Robustheit (niedrigere Varianz = besser)', fontweight='bold')
        ax.set_ylabel('Performance Varianz (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.scenario_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _create_difficulty_panel(self, ax, ql_success: list, dqn_success: list) -> None:
        """Panel D: Szenario-Schwierigkeit."""
        x_pos = np.arange(len(self.scenarios))
        difficulty = [100 - (ql + dqn) / 2 for ql, dqn in zip(ql_success, dqn_success)]

        bars = ax.bar(x_pos, difficulty, alpha=0.7)

        # Farbkodierung nach Schwierigkeit
        for bar, diff in zip(bars, difficulty):
            if diff < 20:
                bar.set_color(self.colors['positive'])
            elif diff < 50:
                bar.set_color(self.colors['neutral'])
            else:
                bar.set_color(self.colors['negative'])

        ax.set_title('D) Szenario-Schwierigkeit', fontweight='bold')
        ax.set_ylabel('Schwierigkeits-Index (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.scenario_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Werte anzeigen
        for i, diff in enumerate(difficulty):
            ax.text(i, diff + 2, f'{diff:.0f}%', ha='center', va='bottom', fontsize=9)

    def _create_stability_panel(self, ax, ql_results: Dict, dqn_results: Dict) -> None:
        """Panel E: Performance Stabilität Heatmap."""
        stability_matrix = []

        for scenario in self.scenarios:
            row = []
            # Q-Learning Segmente
            if scenario in ql_results:
                episodes = ql_results[scenario]['success_per_episode']
                segments = np.array_split(episodes, 3)
                row.extend([np.mean(seg) * 100 for seg in segments])
            else:
                row.extend([0, 0, 0])

            # DQN Segmente
            if scenario in dqn_results:
                episodes = dqn_results[scenario]['success_per_episode']
                segments = np.array_split(episodes, 3)
                row.extend([np.mean(seg) * 100 for seg in segments])
            else:
                row.extend([0, 0, 0])

            stability_matrix.append(row)

        im = ax.imshow(stability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        ax.set_title('E) Performance Stabilität', fontweight='bold')
        ax.set_ylabel('Szenarien')
        ax.set_xlabel('Algorithmus-Segmente')
        ax.set_yticks(range(len(self.scenarios)))
        ax.set_yticklabels(self.scenario_labels)
        ax.set_xticks(range(6))
        ax.set_xticklabels(['QL-1', 'QL-2', 'QL-3', 'DQN-1', 'DQN-2', 'DQN-3'])

        plt.colorbar(im, ax=ax, shrink=0.8, label='Erfolgsrate (%)')

    def _create_superiority_panel(self, ax, ql_success: list, dqn_success: list) -> None:
        """Panel F: Algorithmus-Überlegenheit."""
        x_pos = np.arange(len(self.scenarios))
        margins = []
        colors = []

        for ql, dqn in zip(ql_success, dqn_success):
            margin = abs(ql - dqn)
            margins.append(margin)

            if ql > dqn:
                colors.append(self.colors['qlearning'])
            elif dqn > ql:
                colors.append(self.colors['dqn'])
            else:
                colors.append(self.colors['neutral'])

        ax.bar(x_pos, margins, color=colors, alpha=0.8)

        ax.set_title('F) Algorithmus-Überlegenheit', fontweight='bold')
        ax.set_ylabel('Erfolgsrate-Differenz (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.scenario_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Legende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['qlearning'], label='Q-Learning Superior'),
            Patch(facecolor=self.colors['dqn'], label='DQN Superior'),
            Patch(facecolor=self.colors['neutral'], label='Equivalent')
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

    def _save_plot(self) -> None:
        """Speichert Plot als PDF."""
        save_path = os.path.join("exports", "algorithm_comparison_analysis.pdf")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')


def main():
    """Haupteinstiegspunkt."""
    comparator = AlgorithmComparator()
    comparator.run_comparison()


if __name__ == "__main__":
    main()