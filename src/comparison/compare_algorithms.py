# src/comparison/fair_algorithm_comparison.py

import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ZENTRALE CONFIG - Alle Parameter stammen aus der gemeinsamen config.py
from src.shared.config import (
    SEED, EPISODES, DQN_EPISODES, EVAL_EPISODES, EVAL_MAX_STEPS, MAX_STEPS,
    EXPORT_PATH_COMP, get_dqn_model_path, get_q_table_path,
    GAMMA, ALPHA, EPSILON,  # Q-Learning Parameter
    DQN_LEARNING_RATE, DQN_EPSILON_START, DQN_EPSILON_END, DQN_EPSILON_DECAY  # DQN Parameter
)


@dataclass
class ComparisonConfig:
    """Zentrale Konfiguration für fairen Algorithmus-Vergleich - basiert vollständig auf config.py."""

    # Alle Parameter aus der zentralen config.py
    base_seed: int = SEED
    eval_episodes: int = EVAL_EPISODES
    max_steps_per_episode: int = EVAL_MAX_STEPS  # Verwende EVAL_MAX_STEPS für Evaluation

    # Episodes aus config.py
    q_learning_episodes: int = EPISODES
    dqn_episodes: int = DQN_EPISODES

    # Szenarien
    scenarios: List[str] = None

    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = [
                'static',
                'random_start',
                'random_goal',
                'random_obstacles',
                'container'
            ]

    def get_seed_for_run(self, run: int) -> int:
        """Gibt deterministischen Seed für einen spezifischen Run zurück."""
        return self.base_seed + run

    def validate_and_report_config(self):
        """Validiert und berichtet über die verwendete Konfiguration."""
        print("🔧 FAIRNESS-KONFIGURATION (aus config.py)")
        print("=" * 60)
        print(f"Basis-Seed: {self.base_seed}")
        print(f"Evaluation Episodes: {self.eval_episodes}")
        print(f"Max Steps per Evaluation Episode: {self.max_steps_per_episode}")
        print()
        print("Training-Parameter (für Referenz):")
        print(f"  Q-Learning Episodes (Training): {self.q_learning_episodes}")
        print(f"  DQN Episodes (Training): {self.dqn_episodes}")
        print()
        print("Hyperparameter-Synchronisation:")
        print(f"  Discount Factor (γ): {GAMMA} (beide Algorithmen)")
        print(f"  Q-Learning α: {ALPHA}, ε: {EPSILON}")
        print(f"  DQN Learning Rate: {DQN_LEARNING_RATE}")
        print(f"  DQN ε: {DQN_EPSILON_START} → {DQN_EPSILON_END} (decay: {DQN_EPSILON_DECAY})")
        print()
        print(f"Szenarien: {self.scenarios}")

        # Fairness-Checks
        fairness_issues = []

        if self.q_learning_episodes != self.dqn_episodes:
            fairness_issues.append(
                f"ℹ️  INFO: Unterschiedliche Training-Episodes "
                f"(Q-Learning: {self.q_learning_episodes}, DQN: {self.dqn_episodes})"
            )

        if self.eval_episodes < 100:
            fairness_issues.append(
                f"⚠️  WARNUNG: Niedrige Evaluation-Episodes ({self.eval_episodes}). "
                f"Für robuste Statistiken werden ≥100 empfohlen."
            )

        if fairness_issues:
            print("\nFairness-Hinweise:")
            for issue in fairness_issues:
                print(f"  {issue}")
        else:
            print("\n✅ Alle Fairness-Parameter optimal konfiguriert")

        print("=" * 60)
        print()


@dataclass
class EvaluationResult:
    """Standardisiertes Ergebnis-Format für alle Algorithmen."""
    success_rate: float
    average_steps: float
    average_reward: float
    duration: float
    additional_metrics: Dict[str, Any] = None


class AlgorithmEvaluator(ABC):
    """Abstract Base Class für Algorithmus-Evaluation."""

    def __init__(self, config: ComparisonConfig):
        self.config = config

    @abstractmethod
    def evaluate(self, scenario: str, episodes: int, run: int) -> EvaluationResult:
        """Evaluiert einen trainierten Algorithmus."""
        pass

    @abstractmethod
    def is_model_available(self, scenario: str) -> bool:
        """Prüft ob ein trainiertes Modell verfügbar ist."""
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Gibt den Namen des Algorithmus zurück."""
        pass


class QLearningEvaluator(AlgorithmEvaluator):
    """Evaluator für Q-Learning basierend auf bestehender Infrastruktur."""

    def __init__(self, config: ComparisonConfig):
        super().__init__(config)

        # Import der bestehenden Q-Learning Module
        sys.path.append(os.path.join(os.path.dirname(__file__), "../q_learning"))
        from utils.common import set_all_seeds, obs_to_state, check_success
        from utils.environment import initialize_environment_for_scenario
        from utils.qlearning import load_q_table
        from utils.evaluation import check_loop_detection

        self.set_all_seeds = set_all_seeds
        self.obs_to_state = obs_to_state
        self.check_success = check_success
        self.initialize_environment_for_scenario = initialize_environment_for_scenario
        self.load_q_table = load_q_table
        self.check_loop_detection = check_loop_detection

    def get_algorithm_name(self) -> str:
        return "Q-Learning"

    def is_model_available(self, scenario: str) -> bool:
        """Prüft ob Q-Table für Szenario existiert."""
        q_table_path = get_q_table_path(scenario)
        full_path = os.path.join(os.path.dirname(__file__), "../q_learning", q_table_path)
        return os.path.exists(full_path)

    def evaluate(self, scenario: str, episodes: int, run: int) -> EvaluationResult:
        """Evaluiert Q-Learning Policy mit bestehender Infrastruktur."""
        start_time = time.time()

        # Seed setzen für Reproduzierbarkeit - SYNCHRONISIERT mit DQN
        seed = self.config.get_seed_for_run(run)
        self.set_all_seeds()
        np.random.seed(seed)

        # Szenario-Konfiguration
        scenario_config = {
            "env_mode": scenario,
            "q_table_path": get_q_table_path(scenario),
            "environment": "container" if scenario == "container" else "grid"
        }

        # Environment und Q-Table laden
        env, grid_size = self.initialize_environment_for_scenario(scenario_config)
        Q = self.load_q_table(scenario)

        if Q is None:
            raise FileNotFoundError(f"Q-Table für Szenario '{scenario}' nicht gefunden")

        # Evaluation durchführen mit GLEICHEN Parametern wie DQN
        results = self._run_evaluation_episodes(env, Q, grid_size, scenario, episodes)

        duration = time.time() - start_time

        return EvaluationResult(
            success_rate=results['success_rate'],
            average_steps=results['average_steps'],
            average_reward=results['average_reward'],
            duration=duration,
            additional_metrics={
                'timeout_count': results['timeout_count'],
                'loop_abort_count': results['loop_abort_count'],
                'obstacle_count': results['obstacle_count']
            }
        )

    def _run_evaluation_episodes(self, env, Q, grid_size, scenario, episodes):
        """Führt Evaluation-Episoden durch - identisch zu DQN-Evaluation-Logik."""
        results = {
            "success_count": 0,
            "timeout_count": 0,
            "loop_abort_count": 0,
            "obstacle_count": 0,
            "episode_rewards": [],
            "steps_to_goal": []
        }

        # Verwendet EVAL_MAX_STEPS aus config.py - IDENTISCH zu DQN
        max_steps = self.config.max_steps_per_episode

        for episode in range(episodes):
            obs, _ = env.reset()
            state = self.obs_to_state(obs, scenario, grid_size)
            episode_reward = 0
            steps = 0
            visited_states = {}

            terminated_by_environment = False
            while steps < max_steps:
                action = np.argmax(Q[state])
                obs, reward, terminated, _, _ = env.step(action)
                next_state = self.obs_to_state(obs, scenario, grid_size)
                episode_reward += reward
                steps += 1

                if self.check_success(reward, scenario):
                    results["success_count"] += 1
                    results["steps_to_goal"].append(steps)
                    break

                if self.check_loop_detection(visited_states, next_state, scenario):
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

        # Metriken berechnen - IDENTISCH zu DQN
        success_rate = (results["success_count"] / episodes) * 100
        average_steps = np.mean(results["steps_to_goal"]) if results["steps_to_goal"] else max_steps
        average_reward = np.mean(results["episode_rewards"])

        return {
            'success_rate': success_rate,
            'average_steps': average_steps,
            'average_reward': average_reward,
            'timeout_count': results["timeout_count"],
            'loop_abort_count': results["loop_abort_count"],
            'obstacle_count': results["obstacle_count"]
        }


class DQNEvaluator(AlgorithmEvaluator):
    """Evaluator für DQN - verwendet nur Evaluation, kein Training."""

    def __init__(self, config: ComparisonConfig):
        super().__init__(config)

        # Import der DQN Module
        from src.dqn.train import DQNTrainer
        self.DQNTrainer = DQNTrainer

    def get_algorithm_name(self) -> str:
        return "DQN"

    def is_model_available(self, scenario: str) -> bool:
        """Prüft ob DQN-Modell für Szenario existiert."""
        model_path = get_dqn_model_path(scenario)
        full_path = os.path.join(os.path.dirname(__file__), "../dqn", model_path)
        return os.path.exists(full_path)

    def evaluate(self, scenario: str, episodes: int, run: int) -> EvaluationResult:
        """Evaluiert DQN Policy ohne Neutraining."""
        start_time = time.time()

        # Seed für Reproduzierbarkeit - SYNCHRONISIERT mit Q-Learning
        seed = self.config.get_seed_for_run(run)
        np.random.seed(seed)

        # DQN Trainer erstellen (ohne Training)
        trainer = self.DQNTrainer(env_mode=scenario)

        # Modell laden und evaluieren - verwendet die bestehende evaluate() Methode
        eval_results = trainer.evaluate(episodes=episodes, load_model=True)

        duration = time.time() - start_time

        return EvaluationResult(
            success_rate=eval_results['success_rate'],
            average_steps=eval_results['average_steps'],
            average_reward=eval_results['average_reward'],
            duration=duration,
            additional_metrics={
                'std_steps': eval_results.get('std_steps', 0),
                'std_reward': eval_results.get('std_reward', 0),
                'individual_results': eval_results.get('individual_results', [])
            }
        )


class FairAlgorithmComparison:
    """Hauptklasse für fairen Algorithmus-Vergleich mit Clean Code Prinzipien."""

    def __init__(self, config: ComparisonConfig = None):
        self.config = config or ComparisonConfig()

        # Erstelle Evaluators mit gemeinsamer Config
        self.evaluators = {
            "Q-Learning": QLearningEvaluator(self.config),
            "DQN": DQNEvaluator(self.config)
        }

    def run_fair_comparison(self, episodes_per_algorithm: Dict[str, int] = None,
                            num_runs: int = 3) -> Dict[str, Any]:
        """
        Führt fairen Vergleich zwischen Algorithmen durch.

        Args:
            episodes_per_algorithm: Dict mit Episoden pro Algorithmus (für Evaluation)
            num_runs: Anzahl Wiederholungen pro Algorithmus/Szenario
        """
        # Validiere und berichte Konfiguration
        self.config.validate_and_report_config()

        if episodes_per_algorithm is None:
            episodes_per_algorithm = {
                "Q-Learning": self.config.eval_episodes,
                "DQN": self.config.eval_episodes
            }

        print("🎯 FAIRER ALGORITHMUS-VERGLEICH")
        print("=" * 80)
        print(f"Algorithmen: {list(self.evaluators.keys())}")
        print(f"Szenarien: {len(self.config.scenarios)}")
        print(f"Runs pro Algorithmus/Szenario: {num_runs}")
        print(f"Evaluation Episodes: {episodes_per_algorithm}")
        print()

        # Pre-Check: Sind alle Modelle verfügbar?
        self._check_model_availability()

        all_results = {}
        comparison_data = []

        for scenario in self.config.scenarios:
            print(f"\n{'=' * 60}")
            print(f"SZENARIO: {scenario.upper()}")
            print(f"{'=' * 60}")

            scenario_results = {}

            for algorithm_name, evaluator in self.evaluators.items():
                print(f"\n{'-' * 30}")
                print(f"{algorithm_name.upper()}")
                print(f"{'-' * 30}")

                algorithm_results = []
                episodes = episodes_per_algorithm.get(algorithm_name, self.config.eval_episodes)

                for run in range(num_runs):
                    print(f"\n{algorithm_name} Run {run + 1}/{num_runs}")
                    print(f"  Seed: {self.config.get_seed_for_run(run)}")

                    try:
                        result = evaluator.evaluate(scenario, episodes, run)
                        algorithm_results.append(result)

                        comparison_data.append({
                            'algorithm': algorithm_name,
                            'scenario': scenario,
                            'run': run + 1,
                            'seed': self.config.get_seed_for_run(run),
                            'success_rate': result.success_rate,
                            'avg_steps': result.average_steps,
                            'avg_reward': result.average_reward,
                            'duration_minutes': result.duration / 60,
                            'episodes': episodes
                        })

                        print(f"  Erfolgsrate: {result.success_rate:.1f}%")
                        print(f"  Durchschnittliche Schritte: {result.average_steps:.1f}")
                        print(f"  Belohnung: {result.average_reward:.2f}")
                        print(f"  Dauer: {result.duration / 60:.1f} min")

                    except Exception as e:
                        print(f"  ❌ Fehler bei {algorithm_name}: {e}")
                        continue

                scenario_results[algorithm_name] = algorithm_results

            all_results[scenario] = scenario_results

            # Szenario-Vergleich
            self._print_scenario_comparison(scenario, scenario_results)

        # Gesamtvergleich und Visualisierung
        self._print_overall_comparison(comparison_data)
        self._save_comparison_results(all_results, comparison_data)
        self._create_comparison_visualizations(comparison_data)

        return {
            'detailed_results': all_results,
            'comparison_data': comparison_data,
            'config': self.config
        }

    def _check_model_availability(self):
        """Prüft ob alle benötigten Modelle verfügbar sind."""
        print("🔍 Überprüfe Modell-Verfügbarkeit...")

        missing_models = []
        available_models = []

        for scenario in self.config.scenarios:
            for algorithm_name, evaluator in self.evaluators.items():
                if evaluator.is_model_available(scenario):
                    available_models.append(f"{algorithm_name}: {scenario}")
                else:
                    missing_models.append(f"{algorithm_name}: {scenario}")

        if missing_models:
            print("❌ Fehlende Modelle:")
            for model in missing_models:
                print(f"  - {model}")
            print("\n💡 Verfügbare Modelle:")
            for model in available_models:
                print(f"  ✓ {model}")
            print("\nBitte führen Sie zuerst das Training für fehlende Szenarien durch:")
            print("  Q-Learning: python src/q_learning/train_all_scenarios.py")
            print("  DQN: python src/dqn/train_all_scenarios.py")

            raise FileNotFoundError("Nicht alle Modelle verfügbar")

        print("✅ Alle Modelle verfügbar")
        print(f"  Verfügbare Modelle: {len(available_models)}")
        print()

    def _print_scenario_comparison(self, scenario: str, results: Dict[str, List[EvaluationResult]]):
        """Gibt Vergleich für ein Szenario aus."""
        print(f"\n{'=' * 40}")
        print(f"VERGLEICH {scenario.upper()}")
        print(f"{'=' * 40}")

        for algorithm_name, algorithm_results in results.items():
            if algorithm_results:
                success_rates = [r.success_rate for r in algorithm_results]
                avg_steps = [r.average_steps for r in algorithm_results]
                avg_rewards = [r.average_reward for r in algorithm_results]

                print(f"\n{algorithm_name}:")
                print(f"  Erfolgsrate: {np.mean(success_rates):.1f}% ± {np.std(success_rates):.1f}%")
                print(f"  Schritte: {np.mean(avg_steps):.1f} ± {np.std(avg_steps):.1f}")
                print(f"  Belohnung: {np.mean(avg_rewards):.2f} ± {np.std(avg_rewards):.2f}")

        # Direkter Vergleich
        algorithms = list(results.keys())
        if len(algorithms) >= 2 and all(len(results[algo]) > 0 for algo in algorithms):
            success_rates = {}
            for algo in algorithms:
                success_rates[algo] = np.mean([r.success_rate for r in results[algo]])

            winner = max(success_rates, key=success_rates.get)
            loser = min(success_rates, key=success_rates.get)
            diff = success_rates[winner] - success_rates[loser]

            print(f"\n🏆 Szenario-Gewinner: {winner} (+{diff:.1f}% Erfolgsrate)")

    def _print_overall_comparison(self, comparison_data: List[Dict]):
        """Gibt Gesamtvergleich aus."""
        print(f"\n{'=' * 100}")
        print("GESAMTVERGLEICH ALLER SZENARIEN")
        print(f"{'=' * 100}")

        df = pd.DataFrame(comparison_data)

        # Gruppiere nach Algorithmus und Szenario
        grouped = df.groupby(['algorithm', 'scenario']).agg({
            'success_rate': ['mean', 'std'],
            'avg_steps': ['mean', 'std'],
            'avg_reward': ['mean', 'std'],
            'duration_minutes': ['mean', 'std']
        }).round(2)

        # Tabelle
        print(f"\n{'Algorithmus':<15} {'Szenario':<18} {'Erfolgsrate':<15} {'Schritte':<15} {'Belohnung':<15}")
        print("-" * 85)

        for algorithm in ['Q-Learning', 'DQN']:  # Feste Reihenfolge
            for scenario in self.config.scenarios:
                if (algorithm, scenario) in grouped.index:
                    success_mean = grouped.loc[(algorithm, scenario), ('success_rate', 'mean')]
                    success_std = grouped.loc[(algorithm, scenario), ('success_rate', 'std')]
                    steps_mean = grouped.loc[(algorithm, scenario), ('avg_steps', 'mean')]
                    steps_std = grouped.loc[(algorithm, scenario), ('avg_steps', 'std')]
                    reward_mean = grouped.loc[(algorithm, scenario), ('avg_reward', 'mean')]
                    reward_std = grouped.loc[(algorithm, scenario), ('avg_reward', 'std')]

                    print(f"{algorithm:<15} {scenario:<18} {success_mean:>6.1f}±{success_std:>4.1f}% "
                          f"{steps_mean:>6.1f}±{steps_std:>4.1f} {reward_mean:>6.2f}±{reward_std:>4.2f}")

        # Algorithmus-Durchschnitte
        print(f"\n{'=' * 60}")
        print("GESAMTLEISTUNG (Alle Szenarien)")
        print(f"{'=' * 60}")

        algo_summary = df.groupby('algorithm').agg({
            'success_rate': ['mean', 'std'],
            'avg_steps': ['mean', 'std'],
            'avg_reward': ['mean', 'std']
        }).round(2)

        for algorithm in ['Q-Learning', 'DQN']:
            if algorithm in algo_summary.index:
                print(f"\n{algorithm}:")
                success_mean = algo_summary.loc[algorithm, ('success_rate', 'mean')]
                success_std = algo_summary.loc[algorithm, ('success_rate', 'std')]
                steps_mean = algo_summary.loc[algorithm, ('avg_steps', 'mean')]
                steps_std = algo_summary.loc[algorithm, ('avg_steps', 'std')]
                reward_mean = algo_summary.loc[algorithm, ('avg_reward', 'mean')]
                reward_std = algo_summary.loc[algorithm, ('avg_reward', 'std')]

                print(f"  Durchschnittliche Erfolgsrate: {success_mean:.1f}% ± {success_std:.1f}%")
                print(f"  Durchschnittliche Schritte: {steps_mean:.1f} ± {steps_std:.1f}")
                print(f"  Durchschnittliche Belohnung: {reward_mean:.2f} ± {reward_std:.2f}")

        # Statistischer Vergleich
        print(f"\n{'=' * 60}")
        print("🏆 FINALER VERGLEICH")
        print(f"{'=' * 60}")

        if 'Q-Learning' in df['algorithm'].values and 'DQN' in df['algorithm'].values:
            ql_data = df[df['algorithm'] == 'Q-Learning']
            dqn_data = df[df['algorithm'] == 'DQN']

            ql_overall_success = ql_data['success_rate'].mean()
            dqn_overall_success = dqn_data['success_rate'].mean()

            overall_winner = "DQN" if dqn_overall_success > ql_overall_success else "Q-Learning"
            difference = abs(dqn_overall_success - ql_overall_success)

            print(f"🥇 GESAMTGEWINNER: {overall_winner}")
            print(f"   Unterschied: {difference:.1f} Prozentpunkte")

            # Szenario-spezifische Gewinner
            print(f"\nSzenario-spezifische Gewinner:")
            scenario_wins = {"Q-Learning": 0, "DQN": 0}

            for scenario in self.config.scenarios:
                ql_scenario = ql_data[ql_data['scenario'] == scenario]['success_rate'].mean()
                dqn_scenario = dqn_data[dqn_data['scenario'] == scenario]['success_rate'].mean()

                if not pd.isna(ql_scenario) and not pd.isna(dqn_scenario):
                    scenario_winner = "DQN" if dqn_scenario > ql_scenario else "Q-Learning"
                    scenario_diff = abs(dqn_scenario - ql_scenario)
                    scenario_wins[scenario_winner] += 1
                    print(f"  {scenario:<18}: {scenario_winner} (+{scenario_diff:.1f}%)")

            print(f"\nSzenarien-Bilanz:")
            print(f"  Q-Learning: {scenario_wins['Q-Learning']} Szenarien gewonnen")
            print(f"  DQN: {scenario_wins['DQN']} Szenarien gewonnen")

    def _save_comparison_results(self, all_results: Dict, comparison_data: List[Dict]):
        """Speichert Vergleichsergebnisse."""
        # CSV mit allen Daten
        df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(EXPORT_PATH_COMP, 'fair_algorithm_comparison.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nVergleichsdaten gespeichert: {csv_path}")

        # Zusammenfassungsstatistiken
        summary_stats = df.groupby(['algorithm', 'scenario']).agg({
            'success_rate': ['mean', 'std', 'min', 'max'],
            'avg_steps': ['mean', 'std', 'min', 'max'],
            'avg_reward': ['mean', 'std', 'min', 'max']
        }).round(3)

        stats_path = os.path.join(EXPORT_PATH_COMP, 'fair_algorithm_comparison_stats.csv')
        summary_stats.to_csv(stats_path)
        print(f"Zusammenfassungsstatistiken gespeichert: {stats_path}")

    def _create_comparison_visualizations(self, comparison_data: List[Dict]):
        """Erstellt Vergleichs-Visualisierungen."""
        df = pd.DataFrame(comparison_data)

        # Hauptvergleichs-Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fair Algorithm Comparison: Q-Learning vs Deep Q-Learning\n(Alle Parameter aus config.py)',
                     fontsize=16)

        scenarios = self.config.scenarios
        algorithms = ['Q-Learning', 'DQN']

        x = np.arange(len(scenarios))
        width = 0.35

        # Farben für Algorithmen
        colors = {'Q-Learning': 'skyblue', 'DQN': 'lightcoral'}

        # Success Rate
        for i, algorithm in enumerate(algorithms):
            means = []
            stds = []
            for scenario in scenarios:
                scenario_data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
                if len(scenario_data) > 0:
                    means.append(scenario_data['success_rate'].mean())
                    stds.append(scenario_data['success_rate'].std())
                else:
                    means.append(0)
                    stds.append(0)

            axes[0, 0].bar(x + i * width - width / 2, means, width,
                           yerr=stds, capsize=3, alpha=0.8,
                           color=colors[algorithm], label=algorithm)

        axes[0, 0].set_title('Erfolgsrate')
        axes[0, 0].set_ylabel('Erfolgsrate (%)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scenarios, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Average Steps
        for i, algorithm in enumerate(algorithms):
            means = []
            stds = []
            for scenario in scenarios:
                scenario_data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
                if len(scenario_data) > 0:
                    means.append(scenario_data['avg_steps'].mean())
                    stds.append(scenario_data['avg_steps'].std())
                else:
                    means.append(0)
                    stds.append(0)

            axes[0, 1].bar(x + i * width - width / 2, means, width,
                           yerr=stds, capsize=3, alpha=0.8,
                           color=colors[algorithm], label=algorithm)

        axes[0, 1].set_title('Durchschnittliche Schritte')
        axes[0, 1].set_ylabel('Schritte')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scenarios, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Average Reward
        for i, algorithm in enumerate(algorithms):
            means = []
            stds = []
            for scenario in scenarios:
                scenario_data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
                if len(scenario_data) > 0:
                    means.append(scenario_data['avg_reward'].mean())
                    stds.append(scenario_data['avg_reward'].std())
                else:
                    means.append(0)
                    stds.append(0)

            axes[0, 2].bar(x + i * width - width / 2, means, width,
                           yerr=stds, capsize=3, alpha=0.8,
                           color=colors[algorithm], label=algorithm)

        axes[0, 2].set_title('Durchschnittliche Belohnung')
        axes[0, 2].set_ylabel('Belohnung')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(scenarios, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Box Plots für detaillierte Verteilungen
        # Success Rate Distribution
        success_data = [df[df['algorithm'] == algo]['success_rate'].values for algo in algorithms]
        bp1 = axes[1, 0].boxplot(success_data, labels=algorithms, patch_artist=True)
        for patch, color in zip(bp1['boxes'], [colors[algo] for algo in algorithms]):
            patch.set_facecolor(color)
        axes[1, 0].set_title('Erfolgsrate Verteilung')
        axes[1, 0].set_ylabel('Erfolgsrate (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Steps Distribution
        steps_data = [df[df['algorithm'] == algo]['avg_steps'].values for algo in algorithms]
        bp2 = axes[1, 1].boxplot(steps_data, labels=algorithms, patch_artist=True)
        for patch, color in zip(bp2['boxes'], [colors[algo] for algo in algorithms]):
            patch.set_facecolor(color)
        axes[1, 1].set_title('Schritte Verteilung')
        axes[1, 1].set_ylabel('Schritte')
        axes[1, 1].grid(True, alpha=0.3)

        # Reward Distribution
        reward_data = [df[df['algorithm'] == algo]['avg_reward'].values for algo in algorithms]
        bp3 = axes[1, 2].boxplot(reward_data, labels=algorithms, patch_artist=True)
        for patch, color in zip(bp3['boxes'], [colors[algo] for algo in algorithms]):
            patch.set_facecolor(color)
        axes[1, 2].set_title('Belohnung Verteilung')
        axes[1, 2].set_ylabel('Belohnung')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Speichern
        plot_path = os.path.join(EXPORT_PATH_COMP, 'fair_algorithm_comparison.pdf')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Vergleichs-Plot gespeichert: {plot_path}")
        plt.close()

        # Zusätzlicher Heatmap-Vergleich
        self._create_heatmap_comparison(df)

    def _create_heatmap_comparison(self, df: pd.DataFrame):
        """Erstellt Heatmap-Vergleich."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Algorithmus-Performance Heatmap (Fair Comparison)', fontsize=14)

        scenarios = self.config.scenarios
        algorithms = ['Q-Learning', 'DQN']

        metrics = ['success_rate', 'avg_steps', 'avg_reward']
        titles = ['Erfolgsrate (%)', 'Durchschnittliche Schritte', 'Durchschnittliche Belohnung']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            # Erstelle Matrix
            matrix = []
            for algorithm in algorithms:
                row = []
                for scenario in scenarios:
                    scenario_data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
                    if len(scenario_data) > 0:
                        row.append(scenario_data[metric].mean())
                    else:
                        row.append(0)
                matrix.append(row)

            # Plot Heatmap
            im = axes[idx].imshow(matrix, cmap='RdYlGn' if metric == 'success_rate' else 'viridis',
                                  aspect='auto')

            # Labels
            axes[idx].set_xticks(range(len(scenarios)))
            axes[idx].set_xticklabels(scenarios, rotation=45)
            axes[idx].set_yticks(range(len(algorithms)))
            axes[idx].set_yticklabels(algorithms)
            axes[idx].set_title(title)

            # Werte in Zellen
            for i in range(len(algorithms)):
                for j in range(len(scenarios)):
                    text = axes[idx].text(j, i, f'{matrix[i][j]:.1f}',
                                          ha="center", va="center", color="black", fontweight='bold')

            plt.colorbar(im, ax=axes[idx])

        plt.tight_layout()

        # Speichern
        heatmap_path = os.path.join(EXPORT_PATH_COMP, 'fair_algorithm_heatmap_comparison.pdf')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap-Vergleich gespeichert: {heatmap_path}")
        plt.close()


def main():
    """Hauptfunktion für fairen Algorithmus-Vergleich."""
    import argparse

    parser = argparse.ArgumentParser(description='Fair Q-Learning vs DQN Vergleich')
    parser.add_argument('--eval-episodes', type=int, default=None,
                        help='Anzahl Episoden für Evaluation (Standard: aus config.py)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Anzahl Wiederholungen pro Algorithmus/Szenario')
    parser.add_argument('--scenarios', nargs='+',
                        choices=['static', 'random_start', 'random_goal', 'random_obstacles', 'container'],
                        help='Spezifische Szenarien (Standard: alle)')
    parser.add_argument('--algorithms', nargs='+',
                        choices=['qlearning', 'dqn'], default=['qlearning', 'dqn'],
                        help='Algorithmen für Vergleich')

    args = parser.parse_args()

    # Config erstellen
    config = ComparisonConfig()

    # Szenarien filtern falls spezifiziert
    if args.scenarios:
        config.scenarios = args.scenarios
        print(f"Ausgewählte Szenarien: {args.scenarios}")

    # Episodes konfigurieren
    episodes_config = {}
    if args.eval_episodes:
        episodes_config["Q-Learning"] = args.eval_episodes
        episodes_config["DQN"] = args.eval_episodes

    # Comparison erstellen und ausführen
    comparison = FairAlgorithmComparison(config)

    # Filter Evaluators basierend auf Argumenten
    if set(args.algorithms) != {'qlearning', 'dqn'}:
        filtered_evaluators = {}
        if 'qlearning' in args.algorithms:
            filtered_evaluators["Q-Learning"] = comparison.evaluators["Q-Learning"]
        if 'dqn' in args.algorithms:
            filtered_evaluators["DQN"] = comparison.evaluators["DQN"]
        comparison.evaluators = filtered_evaluators

    start_time = time.time()
    results = comparison.run_fair_comparison(
        episodes_per_algorithm=episodes_config,
        num_runs=args.runs
    )
    total_time = time.time() - start_time

    print(f"\n{'=' * 80}")
    print("✅ FAIRER VERGLEICH ABGESCHLOSSEN")
    print(f"{'=' * 80}")
    print(f"Gesamtdauer: {total_time / 60:.1f} Minuten")
    print(f"Konfiguration: Alle Parameter aus config.py")
    print(f"Reproduzierbarkeit: Base Seed {config.base_seed} + Run-Offset")
    print(f"Datensicherheit: Nur Evaluation trainierter Modelle")

    return results


if __name__ == "__main__":
    main()