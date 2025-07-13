import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
exports_base = os.path.abspath(os.path.join(SCRIPT_DIR, "exports"))
versions = ["v1", "v2"]
scenarios = ["static", "random_start", "random_goal", "random_obstacles", "container"]

output_dir = os.path.join(exports_base, "comparison")
os.makedirs(output_dir, exist_ok=True)

def moving_average(data, window_size):
    if len(data) < window_size:
        return data  # keine Glättung bei zu kurzem Datensatz
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Farbpalette für Q-Learning Versionen (v1 dunkelblau, v2 hellblau)
colors = {
    "v1": "#003366",  # Dunkelblau
    "v2": "#66B2FF",  # Hellblau
}

def plot_metric_comparison(metric_name, output_filename):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        for version in versions:
            file_path = os.path.join(exports_base, version, f"{metric_name}_{scenario}.npy")
            if os.path.isfile(file_path):
                data = np.load(file_path)
                if metric_name == "learning_curve":
                    window_size = 20
                elif metric_name == "success_curve":
                    window_size = min(max(10, 500 // 20), len(data) // 10)
                else:
                    window_size = 20

                smoothed_data = moving_average(data, window_size)
                x_vals = range(window_size - 1, window_size - 1 + len(smoothed_data))
                ax.plot(x_vals, smoothed_data, label=f"{version.upper()} (MA {window_size})",
                        linewidth=2, color=colors[version])
            else:
                ax.text(0.5, 0.5, f"{version} fehlt", transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='red')

        ax.set_title(scenario.replace("_", " ").capitalize())
        ax.set_xlabel("Episode")
        ax.set_ylabel("Wert")
        ax.grid(True, alpha=0.3)
        ax.legend()

    if len(scenarios) < len(axes):
        for i in range(len(scenarios), len(axes)):
            fig.delaxes(axes[i])

    # tight_layout aufrufen (Platz zwischen Untertitel und Plots)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # Haupttitel
    title_map = {
        "learning_curve": "Learning Curves (Q-Learning)",
        "success_curve": "Success Rate Curves (Q-Learning)"
    }
    main_title = title_map.get(metric_name, metric_name.replace('_', ' ').capitalize())
    fig.suptitle(main_title, fontsize=16, y=0.95)

    # Untertitel mit kleinerer Schrift - näher ran
    subtitle = "Hyperparameter Setup v1 vs. v2"
    fig.text(0.5, 0.91, subtitle, ha='center', va='center', fontsize=12)

    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Vergleichsplot gespeichert: {save_path}")

if __name__ == "__main__":
    plot_metric_comparison("learning_curve", "comparison_learning_curves.pdf")
    plot_metric_comparison("success_curve", "comparison_success_curves.pdf")
