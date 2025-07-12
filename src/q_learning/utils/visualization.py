# utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.shared.config import EXPORT_PDF, EXPORT_PATH_QL, EPISODES


# Erstellung des Export-Ordners
def setup_export():
    if EXPORT_PDF:
        Path(EXPORT_PATH_QL).mkdir(exist_ok=True)


# ============================================================================
# Training Visualizations (aus train.py)
# ============================================================================

# Erstellung der Lernkurve mit Moving Average
def create_learning_curve(rewards_per_episode, env_mode, window_size=20, show=True):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode, alpha=0.3, label="Raw Reward", color='blue')

    if len(rewards_per_episode) >= window_size:
        moving_avg = np.convolve(rewards_per_episode, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards_per_episode)), moving_avg,
                 label=f"Moving Average ({window_size})", color='red', linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Gesamtreward")
    plt.title(f"Lernkurve ({env_mode}-Modus)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if EXPORT_PDF:
        filename = f"{EXPORT_PATH_QL}/train_learning_curve_{env_mode}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Learning Curve gespeichert: {filename}")

    if show:
        plt.show()
    else:
        plt.close()


# Darstellung der Zielerreichung pro Episode
def create_success_curve(success_per_episode, env_mode, show=True):
    plt.figure(figsize=(12, 4))
    plt.plot(success_per_episode, label="Ziel erreicht", color='green', alpha=0.7, linewidth=1)

    window_size = min(max(10, EPISODES // 20), len(success_per_episode) // 10)
    if len(success_per_episode) >= window_size:
        success_moving_avg = np.convolve(success_per_episode, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(success_per_episode)), success_moving_avg,
                 label=f"Erfolgsrate MA ({window_size})", color='darkgreen', linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Erfolg (0/1)")
    plt.title(f"Zielerreichung pro Episode ({env_mode}-Modus)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if EXPORT_PDF:
        filename = f"{EXPORT_PATH_QL}/train_success_curve_{env_mode}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Success Curve gespeichert: {filename}")

    if show:
        plt.show()
    else:
        plt.close()


# Zusammenstellung mehrerer Statistiken (Histogramm, Erfolgsrate, letzte Rewards, Erfolg/Misserfolg)
def create_training_statistics(rewards_per_episode, success_per_episode, env_mode, show=True):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.hist(rewards_per_episode, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_title("Verteilung der Episode-Rewards")
    ax1.set_xlabel("Reward")
    ax1.set_ylabel("Häufigkeit")
    ax1.axvline(np.mean(rewards_per_episode), color='red', linestyle='--',
                label=f'Durchschnitt: {np.mean(rewards_per_episode):.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    cumulative_success = np.cumsum(success_per_episode) / np.arange(1, len(success_per_episode) + 1)
    ax2.plot(cumulative_success, color='green', linewidth=2)
    ax2.set_title("Kumulative Erfolgsrate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Erfolgsrate")
    ax2.grid(True, alpha=0.3)

    display_episodes = min(1000, len(rewards_per_episode) // 2)
    ax3.plot(rewards_per_episode[-display_episodes:], alpha=0.6, color='blue')
    ax3.set_title(f"Reward-Entwicklung (letzte {display_episodes} Episoden)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Reward")
    ax3.grid(True, alpha=0.3)

    total_success = sum(success_per_episode)
    total_failure = len(success_per_episode) - total_success
    ax4.bar(['Erfolg', 'Misserfolg'], [total_success, total_failure],
            color=['green', 'red'], alpha=0.7)
    ax4.set_title("Erfolg vs. Misserfolg")
    ax4.set_ylabel("Anzahl Episoden")

    for i, v in enumerate([total_success, total_failure]):
        percentage = (v / len(success_per_episode)) * 100
        ax4.text(i, v + len(success_per_episode) * 0.01, f'{v}\n({percentage:.1f}%)',
                 ha='center', va='bottom')

    plt.suptitle(f"Trainingsstatistiken ({env_mode}-Modus)", fontsize=16)
    plt.tight_layout()

    if EXPORT_PDF:
        filename = f"{EXPORT_PATH_QL}/train_statistics_{env_mode}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Training Statistics gespeichert: {filename}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# Evaluation Visualizations (aus evaluate_policy.py)
# ============================================================================

def create_success_plot(results_solved, env_mode):
    plt.figure(figsize=(8, 5))
    colors = []
    labels = list(results_solved.keys())
    for label in labels:
        if "solved" in label.lower() or "success" in label.lower():
            colors.append("green")
        else:
            colors.append("red")

    bars = plt.bar(results_solved.keys(), results_solved.values(), color=colors)
    plt.title(f"Lösungsrate ({env_mode}-Modus)")
    plt.xlabel("Ergebnis")
    plt.ylabel("Anzahl Episoden")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()

    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH_QL}/evaluate_policy_success_rate.pdf", format='pdf', bbox_inches='tight')
        print(f"Success Rate Plot gespeichert: {EXPORT_PATH_QL}/evaluate_policy_success_rate.pdf")

    plt.show()


def create_reward_histogram(rewards_all, env_mode):
    avg_reward = np.mean(rewards_all)

    plt.figure(figsize=(10, 6))
    plt.hist(rewards_all, bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Verteilung der Episoden-Rewards ({env_mode}-Modus)")
    plt.xlabel("Kumulativer Episode-Reward")
    plt.ylabel("Häufigkeit")
    plt.axvline(avg_reward, color='red', linestyle='--',
                label=f'Durchschnitt: {avg_reward:.2f}')
    plt.legend()
    plt.tight_layout()

    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH_QL}/evaluate_policy_reward_histogram.pdf", format='pdf', bbox_inches='tight')
        print(f"Reward Histogram gespeichert: {EXPORT_PATH_QL}/evaluate_policy_reward_histogram.pdf")

    plt.show()


# ============================================================================
# Comparison Visualizations (aus legacy_compare_scenarios.py)
# ============================================================================

def create_comparison_table(all_metrics):
    data = []
    for scenario_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        data.append({
            "Szenario": scenario_name,
            "Erfolg (%)": f"{metrics['success_rate'] * 100:.1f}",
            "Timeout (%)": f"{metrics['timeout_rate'] * 100:.1f}",
            "Schleifen (%)": f"{metrics['loop_abort_rate'] * 100:.1f}",
            "Hindernisse (%)": f"{metrics['obstacle_rate'] * 100:.1f}",
            "Ø Reward": f"{metrics['avg_reward']:.2f}",
            "Ø Schritte": f"{metrics['avg_steps_to_goal']:.1f}" if metrics['avg_steps_to_goal'] else "N/A"
        })

    df = pd.DataFrame(data)
    print("\n" + "=" * 80)
    print("SZENARIEN-VERGLEICH")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    return df


def create_success_rate_comparison(all_metrics):
    scenarios = [name for name, metrics in all_metrics.items() if metrics is not None]
    success_rates = [metrics["success_rate"] * 100 for name, metrics in all_metrics.items() if metrics is not None]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios, success_rates, color='lightgray', edgecolor='black')

    plt.title("Erfolgsraten-Vergleich", fontweight='bold', pad=20)
    plt.xlabel("Szenario")
    plt.ylabel("Erfolgsrate (%)")
    plt.ylim(0, 110)

    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH_QL}/success_rates.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def create_stacked_failure_chart(all_metrics):
    scenarios = []
    success_rates = []
    timeout_rates = []
    loop_rates = []
    obstacle_rates = []

    for scenario_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        scenarios.append(scenario_name)
        success_rates.append(metrics["success_rate"] * 100)
        timeout_rates.append(metrics["timeout_rate"] * 100)
        loop_rates.append(metrics["loop_abort_rate"] * 100)
        obstacle_rates.append(metrics["obstacle_rate"] * 100)

    fig, ax = plt.subplots(figsize=(12, 8))

    bottom_timeout = success_rates
    bottom_loop = [success_rates[i] + timeout_rates[i] for i in range(len(scenarios))]
    bottom_obstacle = [success_rates[i] + timeout_rates[i] + loop_rates[i] for i in range(len(scenarios))]

    ax.bar(scenarios, success_rates, label='Erfolg', color='green', alpha=0.8)
    ax.bar(scenarios, timeout_rates, bottom=bottom_timeout, label='Timeout', color='red', alpha=0.7)
    ax.bar(scenarios, loop_rates, bottom=bottom_loop, label='Schleifenabbruch', color='orange', alpha=0.7)
    ax.bar(scenarios, obstacle_rates, bottom=bottom_obstacle, label='Hindernis-Kollision', color='brown', alpha=0.7)

    ax.set_xlabel('Szenario')
    ax.set_ylabel('Anteil (%)')
    ax.set_title('Terminierungsarten pro Szenario', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45)
    plt.tight_layout()

    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH_QL}/failure_modes.pdf", format='pdf', bbox_inches='tight')
    plt.show()
