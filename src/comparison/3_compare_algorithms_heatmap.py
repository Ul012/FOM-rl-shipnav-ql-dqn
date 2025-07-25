# src/comparison/3_compare_algorithms_heatmap.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.shared.config import EXPORT_PATH_COMP, SETUP_NAME
from src.shared.config_utils import get_export_path


def create_heatmap_visualization(comparison_data):
    """Erstellt Heatmap Szenario-Vergleich."""

    df = pd.DataFrame(comparison_data)
    colors = {'Q-Learning': '#1f77b4', 'DQN': '#ff7f0e'}

    # Setup - Heatmap Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Single-Agent Szenario-Vergleich', fontsize=16, fontweight='bold')

    # Ohne Container für bessere Darstellung
    scenarios = ['static', 'random_start', 'random_obstacles', 'random_goal', 'container']
    scenario_labels = ['Static', 'Random Start', 'Random Obstacles', 'Random Goal', 'Container']
    algorithms = ['Q-Learning', 'DQN']
    x = np.arange(len(scenario_labels))
    width = 0.35

    # Helper function
    def get_scenario_data(algorithm, scenarios, metric):
        means, stds = [], []
        for scenario in scenarios:
            data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
            means.append(data[metric].mean() if len(data) > 0 else 0)
            stds.append(data[metric].std() if len(data) > 1 else 0)
        return means, stds

    # Panel 1: Erfolgsrate nach Szenario
    ax1 = axes[0, 0]
    for i, algorithm in enumerate(algorithms):
        means, stds = get_scenario_data(algorithm, scenarios, 'success_rate')
        ax1.bar(x + i * width - width / 2, means, width,
                yerr=stds, capsize=3, alpha=0.8,
                color=colors[algorithm], label=algorithm)

    ax1.set_title('Erfolgsrate nach Szenario')
    ax1.set_ylabel('Erfolgsrate (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 110)

    # Panel 2: Schritte bis Ziel
    ax2 = axes[0, 1]
    for i, algorithm in enumerate(algorithms):
        means, stds = get_scenario_data(algorithm, scenarios, 'avg_steps')
        ax2.bar(x + i * width - width / 2, means, width,
                yerr=stds, capsize=3, alpha=0.8,
                color=colors[algorithm], label=algorithm)

    ax2.set_title('Schritte bis Ziel')
    ax2.set_ylabel('Durchschnittliche Schritte')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_labels, rotation=45)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Belohnung pro Episode
    ax3 = axes[1, 0]
    for i, algorithm in enumerate(algorithms):
        means, stds = get_scenario_data(algorithm, scenarios, 'avg_reward')
        ax3.bar(x + i * width - width / 2, means, width,
                yerr=stds, capsize=3, alpha=0.8,
                color=colors[algorithm], label=algorithm)

    ax3.set_title('Belohnung pro Episode')
    ax3.set_ylabel('Durchschnittliche Belohnung')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenario_labels, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Erfolgsrate Heatmap
    ax4 = axes[1, 1]
    heatmap_data = []
    for algorithm in algorithms:
        row = []
        for scenario in scenarios:
            data = df[(df['algorithm'] == algorithm) & (df['scenario'] == scenario)]
            row.append(data['success_rate'].mean() if len(data) > 0 else 0)
        heatmap_data.append(row)

    heatmap_array = np.array(heatmap_data)
    im = ax4.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax4.set_xticks(range(len(scenario_labels)))
    ax4.set_xticklabels(scenario_labels, rotation=45)
    ax4.set_yticks(range(len(algorithms)))
    ax4.set_yticklabels(algorithms)
    ax4.set_title('Erfolgsrate Heatmap')

    # Werte in Zellen
    for i in range(len(algorithms)):
        for j in range(len(scenarios)):
            ax4.text(j, i, f'{heatmap_array[i, j]:.1f}%',
                     ha="center", va="center",
                     color="black", fontweight='bold')

    plt.colorbar(im, ax=ax4, shrink=0.8, label='Erfolgsrate (%)')
    plt.tight_layout()

    # Speichern
    save_path = os.path.join(get_export_path(EXPORT_PATH_COMP), f'{SETUP_NAME}algorithm_comparison_heatmap.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Heatmap Vergleich gespeichert: {save_path}")
    plt.close()


def main():
    # Load data from CSV if available
    csv_path = os.path.join(get_export_path(EXPORT_PATH_COMP), 'algorithm_comparison_overview.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        comparison_data = df.to_dict('records')
        create_heatmap_visualization(comparison_data)
    else:
        print("❌ Keine Daten gefunden. Führen Sie zuerst compare_algorithms_overview.py aus.")


if __name__ == "__main__":
    main()