# src/comparison/compare_algorithms_2x2_v1.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.shared.config import EXPORT_PATH_COMP, SETUP_NAME


def create_2x2_v1_visualization(comparison_data):
    """Erstellt wissenschaftliches 2x2 Grid (früher V2)."""

    print("🎨 Erstelle 2x2 V1 Visualisierung...")

    df = pd.DataFrame(comparison_data)
    colors = {'Q-Learning': '#1f77b4', 'DQN': '#ff7f0e'}

    print(f"   Daten geladen: {len(df)} Einträge")

    # Setup
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Algorithm Performance Comparison: Q-Learning vs Deep Q-Learning',
                 fontsize=16, fontweight='bold', y=0.95)

    # Data preparation
    scenarios = df['scenario'].unique()
    scenario_labels = [s.replace('_', ' ').title() for s in scenarios]
    algorithms = ['Q-Learning', 'DQN']
    x_pos = np.arange(len(scenarios))
    width = 0.35

    print(f"   Szenarien: {list(scenarios)}")
    print(f"   Algorithmen: {algorithms}")

    # Helper function
    def get_metric_data(algorithm, scenarios, metric):
        means, stds = [], []
        for scenario in scenarios:
            data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
            means.append(data[metric].mean() if len(data) > 0 else 0)
            stds.append(data[metric].std() if len(data) > 1 else 0)
        return means, stds

    print("   Erstelle Panel A: Success Rate...")
    # Panel A: Success Rate
    ax_a = axes[0, 0]
    for i, algorithm in enumerate(algorithms):
        means, stds = get_metric_data(algorithm, scenarios, 'success_rate')
        ax_a.bar(x_pos + i * width - width / 2, means, width,
                 yerr=stds, capsize=3, alpha=0.8,
                 color=colors[algorithm], label=algorithm)

    ax_a.set_title('A) Success Rate Comparison', fontweight='bold', fontsize=12)
    ax_a.set_ylabel('Success Rate (%)', fontsize=11)
    ax_a.set_xlabel('Scenarios', fontsize=11)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(scenario_labels, rotation=45, ha='right')
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    ax_a.set_ylim(0, 105)

    print("   Erstelle Panel B: Total Reward...")
    # Panel B: Total Reward
    ax_b = axes[0, 1]
    for i, algorithm in enumerate(algorithms):
        means, stds = get_metric_data(algorithm, scenarios, 'avg_reward')
        ax_b.bar(x_pos + i * width - width / 2, means, width,
                 yerr=stds, capsize=3, alpha=0.8,
                 color=colors[algorithm], label=algorithm)

    ax_b.set_title('B) Average Total Reward', fontweight='bold', fontsize=12)
    ax_b.set_ylabel('Average Total Reward', fontsize=11)
    ax_b.set_xlabel('Scenarios', fontsize=11)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(scenario_labels, rotation=45, ha='right')
    ax_b.legend(loc='upper right', fontsize=10)
    ax_b.grid(True, alpha=0.3)

    print("   Erstelle Panel C: Average Steps...")
    # Panel C: Average Steps
    ax_c = axes[1, 0]
    for i, algorithm in enumerate(algorithms):
        means, stds = get_metric_data(algorithm, scenarios, 'avg_steps')
        ax_c.bar(x_pos + i * width - width / 2, means, width,
                 yerr=stds, capsize=3, alpha=0.8,
                 color=colors[algorithm], label=algorithm)

    ax_c.set_title('C) Average Steps to Goal', fontweight='bold', fontsize=12)
    ax_c.set_ylabel('Average Steps', fontsize=11)
    ax_c.set_xlabel('Scenarios', fontsize=11)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(scenario_labels, rotation=45, ha='right')
    ax_c.legend()
    ax_c.grid(True, alpha=0.3)

    print("   Erstelle Panel D: Sample Consistency...")
    # Panel D: Sample Efficiency
    ax_d = axes[1, 1]

    # Collect data for scatter plot
    for algorithm in algorithms:
        success_rates = []
        avg_steps = []
        scenario_names = []

        for scenario in scenarios:
            data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
            if len(data) > 0:
                success_rates.append(data['success_rate'].mean())
                avg_steps.append(data['avg_steps'].mean())
                scenario_names.append(scenario)

        # Create scatter plot
        ax_d.scatter(avg_steps, success_rates,
                     color=colors[algorithm], s=100, alpha=0.7,
                     label=algorithm, edgecolors='black', linewidth=1)

        # Add scenario labels
        label_mapping = {
            'static': 'static',
            'random_start': 'r_start',
            'random_goal': 'r_goal',
            'random_obstacles': 'r_obs',
            'container': 'cont'
        }

        for x, y, name in zip(avg_steps, success_rates, scenario_names):
            label = label_mapping.get(name, name)

            # Position labels to the right and slightly below for Q-Learning, left and above for DQN
            if algorithm == 'Q-Learning':
                offset_x = 2.0  # Rechts neben dem Punkt
                offset_y = -3.0  # Etwas darunter
                ha = 'left'
            else:  # DQN
                offset_x = -2.0  # Links neben dem Punkt
                offset_y = 3.0  # Etwas darüber
                ha = 'right'

            ax_d.annotate(label, (x, y),
                          xytext=(offset_x, offset_y),
                          textcoords='offset points',
                          fontsize=10, alpha=0.9,
                          ha=ha, va='center')


    # Add efficiency zones
    ax_d.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='High Performance (>80%)')
    ax_d.axvline(x=20, color='blue', linestyle='--', alpha=0.5, label='Efficient (<20 steps)')

    ax_d.set_title('D) Sample Efficiency Analysis', fontweight='bold', fontsize=12)
    ax_d.set_xlabel('Average Steps to Goal', fontsize=11)
    ax_d.set_ylabel('Success Rate (%)', fontsize=11)
    ax_d.legend(loc='upper right', fontsize=10)
    ax_d.grid(True, alpha=0.3)
    ax_d.set_xlim(0, max(df['avg_steps']) * 1.1 if len(df) > 0 else 50)
    ax_d.set_ylim(0, 105)

    print("   Finalisiere Layout...")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    save_path = os.path.join(EXPORT_PATH_COMP, f'{SETUP_NAME}algorithm_comparison_2x2_Visual1.pdf')
    print(f"   Speichere: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 2x2 V1 Vergleich gespeichert: {save_path}")
    plt.close()


def main():
    print("🚀 Starte 2x2 V1 Algorithmus-Vergleich")
    print("=" * 50)

    # Load data from CSV if available
    csv_path = os.path.join(EXPORT_PATH_COMP, 'algorithm_comparison_2x3.csv')
    print(f"📁 Suche Daten in: {csv_path}")

    if os.path.exists(csv_path):
        print("✅ CSV-Datei gefunden, lade Daten...")
        df = pd.read_csv(csv_path)
        comparison_data = df.to_dict('records')
        print(f"📊 {len(comparison_data)} Datenpunkte geladen")

        create_2x2_v1_visualization(comparison_data)

        print("=" * 50)
        print("🎯 2x2 V1 Vergleich abgeschlossen!")
    else:
        print("❌ Keine Daten gefunden. Führen Sie zuerst compare_algorithms_2x3.py aus.")
        print("💡 Ausführen mit: python src/comparison/compare_algorithms_2x3.py")


if __name__ == "__main__":
    main()