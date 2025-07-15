# utils/evaluation_export.py

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.shared.config import SETUP_NAME

def export_results_to_csv(results, output_path="exports/evaluation_summary.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Szenario", "Success Rate (%)", "Total Reward", "Avg. Reward", "Avg. Steps", "Reward Variance"])
        for result in results:
            writer.writerow([
                result.get("name", ""),
                result.get("success_rate", ""),
                result.get("total_reward", ""),
                result.get("avg_reward", ""),
                result.get("avg_steps", ""),
                result.get("reward_variance", "")
            ])
    print(f"📄 CSV-Datei erfolgreich gespeichert unter: {output_path}")


def moving_average(values, window_size=25):
    """Berechnet den gleitenden Durchschnitt."""
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')


def create_combined_curve_pdf(scenarios, export_dir, metric="learning"):
    """
    Erstellt eine geglättete kombinierte PDF-Grafik für mehrere Szenarien auf Basis gespeicherter .npy-Dateien.

    Args:
        scenarios (List[str]): Liste der Szenario-Namen.
        export_dir (str): Verzeichnis, in dem die .npy-Dateien liegen und die PDF gespeichert wird.
        metric (str): "learning" oder "success"
    """
    plt.figure(figsize=(10, 6))
    npy_files_to_delete = []

    for scenario in scenarios:
        filename = f"{metric}_curve_{scenario}.npy"
        path = os.path.join(export_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️  Datei nicht gefunden: {path}")
            continue
        data = np.load(path)
        if len(data) < 5:
            continue

        smoothed = moving_average(data, window_size=50)
        label = scenario.replace("_", " ").capitalize()
        plt.plot(smoothed, label=label, linewidth=1.2, alpha=0.8)
        npy_files_to_delete.append(path)

    title = "Learning Curve (Q-Learning)" if metric == "learning" else "Success Rate Curve (Q-Learning)"
    ylabel = "Reward" if metric == "learning" else "Success Rate"

    # Referenzlinie
    if metric == "learning":
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    # elif metric == "success":
    #    plt.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8)

    plt.title(title, fontweight='bold')
    plt.xlabel("Episode", fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Plotbereich geht nur bis 85% Breite
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # PDF speichern in exports/combined
    combined_dir = os.path.join(export_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    filename_out = os.path.join(combined_dir, f"{SETUP_NAME}_scenario-comparison_{metric}_curve.pdf")

    with PdfPages(filename_out) as pdf:
        pdf.savefig()
    plt.close()

    print(f"📈 Kombinierte {title} exportiert: {filename_out}")

    # Temporäre .npy-Dateien löschen - Auskommentiert, da npy Dateien für compare_versioned_curves Visualierung benötigt werden
    # for file_path in npy_files_to_delete:
    #    try:
    #        os.remove(file_path)
    #    except Exception as e:
    #        print(f"⚠️  Fehler beim Löschen von {file_path}: {e}")
