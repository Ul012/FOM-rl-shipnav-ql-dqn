# # src/q_learning/train_all_scenarios.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os
import time
import subprocess
from datetime import datetime

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.config import EXPORT_PATH_QL, SETUP_NAME
from src.shared.config_utils import prepare_export_dirs
prepare_export_dirs()

from utils.common import setup_export
from utils.evaluation_export import export_results_to_csv, create_combined_curve_pdf

# ============================================================================
# Konfiguration
# ============================================================================

SCENARIOS = {
    "static": {
        "env_mode": "static",
        "environment": "grid",
        "description": "Statisches Grid (feste Start-, Ziel- und Hindernis-Positionen)"
    },
    "random_start": {
        "env_mode": "random_start",
        "environment": "grid",
        "description": "Zufällige Startposition"
    },
    "random_goal": {
        "env_mode": "random_goal",
        "environment": "grid",
        "description": "Zufällige Zielposition"
    },
    "random_obstacles": {
        "env_mode": "random_obstacles",
        "environment": "grid",
        "description": "Zufällige Hindernis-Positionen"
    },
    "container": {
        "env_mode": "container",
        "environment": "container",
        "description": "Container-Schiff Umgebung mit Pickup/Dropoff"
    }
}

SHOW_VISUALIZATIONS = False
PARALLEL_TRAINING = False

# ============================================================================
# Ausführung des Trainings für ein einzelnes Szenario
# ============================================================================

def run_training_for_scenario(scenario_name, scenario_config):
    print(f"\n{'=' * 60}")
    print(f"STARTE TRAINING: {scenario_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Beschreibung: {scenario_config['description']}")
    print(f"Modus: {scenario_config['env_mode']}")
    print(f"Umgebung: {scenario_config['environment']}")

    start_time = time.time()

    try:
        # Übergabe von Umgebungsvariablen zur Steuerung des Trainings
        env = os.environ.copy()
        env["ENV_MODE"] = scenario_config["env_mode"]
        env["EXPORT_PDF"] = "False" if not SHOW_VISUALIZATIONS else "True"
        env["SHOW_VISUALIZATIONS"] = "False" if not SHOW_VISUALIZATIONS else "True"
        env["SETUP_NAME"] = SETUP_NAME

        # Aufruf des Trainingsscripts mit Übergabe der Konfiguration
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=1800,
            env=env
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Training erfolgreich abgeschlossen ({duration:.1f}s)")
            print(f"Q-Tabelle gespeichert: q_table_{scenario_config['env_mode']}.npy")
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"❌ Training fehlgeschlagen ({duration:.1f}s)")
            print(f"Fehler: {result.stderr}")

        return result.returncode == 0, result.stdout

    except subprocess.TimeoutExpired:
        print(f"⏰ Training-Timeout nach 30min")
        return False, ""
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return False, ""

# ============================================================================
# Hauptfunktion zur Ausführung aller Szenarien
# ============================================================================

def train_all_scenarios():
    print("🏗️  MULTI-SZENARIO TRAINING")
    print(f"Anzahl Szenarien: {len(SCENARIOS)}")
    print(f"Training-Modus: {'Parallel' if PARALLEL_TRAINING else 'Sequenziell'}")

    # Sicherstellen, dass Exportordner vorhanden ist
    setup_export()

    if PARALLEL_TRAINING:
        print("⚠️  Parallel Training ist derzeit deaktiviert")
        return []

    results = {}
    scenario_results = []

    for i, (scenario_name, scenario_config) in enumerate(SCENARIOS.items(), 1):
        print(f"\n[{i}/{len(SCENARIOS)}] Nächstes Szenario: {scenario_name}")
        success, stdout = run_training_for_scenario(scenario_name, scenario_config)

        results[scenario_name] = success
        time.sleep(2)

        # Hilfsfunktion zum Extrahieren numerischer Werte aus der Konsolenausgabe
        def extract_value(label, text, cast_fn=float):
            for line in text.splitlines():
                if label in line:
                    try:
                        return cast_fn(line.split(":")[1].split()[0])
                    except (IndexError, ValueError):
                        return None
            return None

        # Sammeln der Metriken pro Szenario
        scenario_results.append({
            "name": scenario_name,
            "success_rate": extract_value("Erfolgreiche Episoden", stdout,
                                          lambda x: float(x.split("/")[0]) / float(x.split("/")[1].strip("()")) * 100),
            "total_reward": extract_value("Gesamtreward", stdout),
            "avg_reward": extract_value("Durchschnitt", stdout),
            "avg_steps": extract_value("Durchschnittliche Schritte", stdout),
            "reward_variance": extract_value("Varianz", stdout)
        })

    print(f"\n{'=' * 60}")
    print("TRAINING ZUSAMMENFASSUNG")
    print(f"{'=' * 60}")

    # Zusammenfassung in Konsole
    for scenario_name, success in results.items():
        status = "✅ Erfolgreich" if success else "❌ Fehlgeschlagen"
        q_table_exists = os.path.exists(f"q_table_{SCENARIOS[scenario_name]['env_mode']}.npy")
        q_table_status = "Q-Tabelle ✓" if q_table_exists else "Q-Tabelle ✗"
        print(f"{scenario_name:<20} {status:<15} {q_table_status}")

    # Export als CSV-Datei
    csv_path = os.path.join(EXPORT_PATH_QL, SETUP_NAME, "evaluation_summary.csv")
    export_results_to_csv(scenario_results, output_path=csv_path)

    return scenario_results

# ============================================================================
# Einstiegspunkt
# ============================================================================

if __name__ == "__main__":
    scenario_results = train_all_scenarios()

    # Erzeugung kombinierter PDF-Grafiken für Lernverlauf und Erfolgsquote
    scenario_names = [result["name"] for result in scenario_results]
    export_dir = os.path.join(os.path.dirname(__file__), "exports", SETUP_NAME)
    create_combined_curve_pdf(scenario_names, export_dir=f"exports/{SETUP_NAME}", metric="learning")
    create_combined_curve_pdf(scenario_names, export_dir=f"exports/{SETUP_NAME}", metric="success")
