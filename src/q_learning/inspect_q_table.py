# inspect_q_tables.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Drittanbieter
import numpy as np

# Lokale Module
from src.shared.config import ENV_MODE

# Utils

# ============================================================================
# Konfiguration
# ============================================================================

# Verfügbare Q-Tabellen
AVAILABLE_Q_TABLES = {
    "static": "q_table_static.npy",
    "random_start": "q_table_random_start.npy",
    "random_goal": "q_table_random_goal.npy",
    "random_obstacles": "q_table_random_obstacles.npy",
    "container": "q_table_container.npy"
}

# Anzahl der anzuzeigenden Einträge
DISPLAY_ROWS = 5
DISPLAY_COLS = 4


# ============================================================================
# Hilfsfunktionen
# ============================================================================

def load_q_table_for_inspection(filepath):
    """Q-Tabelle für Inspektion laden"""
    try:
        q_table = np.load(filepath)
        print(f"Q-Tabelle erfolgreich geladen: {filepath}")
        return q_table
    except FileNotFoundError:
        print(f"FEHLER: Q-Tabelle nicht gefunden: {filepath}")
        return None


def get_q_table_info(q_table):
    """Informationen über Q-Tabelle sammeln"""
    if q_table is None:
        return None

    info = {
        "shape": q_table.shape,
        "total_states": q_table.shape[0],
        "total_actions": q_table.shape[1],
        "min_value": np.min(q_table),
        "max_value": np.max(q_table),
        "mean_value": np.mean(q_table),
        "std_value": np.std(q_table),
        "zero_entries": np.sum(q_table == 0),
        "nonzero_entries": np.sum(q_table != 0)
    }

    return info


def find_best_actions_per_state(q_table, num_states=10):
    """Beste Aktionen für erste N Zustände finden"""
    if q_table is None:
        return None

    best_actions = []
    for state in range(min(num_states, q_table.shape[0])):
        best_action = np.argmax(q_table[state])
        best_value = q_table[state, best_action]
        best_actions.append((state, best_action, best_value))

    return best_actions


# ============================================================================
# Anzeigefunktionen
# ============================================================================

def display_q_table_overview(q_table, scenario_name):
    """Übersicht über Q-Tabelle anzeigen"""
    if q_table is None:
        print(f"Keine Q-Tabelle für Szenario '{scenario_name}' verfügbar.")
        return

    print(f"\n{'=' * 60}")
    print(f"Q-TABELLE ÜBERSICHT: {scenario_name.upper()}")
    print(f"{'=' * 60}")

    # Grundlegende Informationen
    info = get_q_table_info(q_table)
    print(f"Form (States x Actions): {info['shape']}")
    print(f"Gesamte Zustände: {info['total_states']}")
    print(f"Gesamte Aktionen: {info['total_actions']}")
    print(f"Min/Max Werte: {info['min_value']:.3f} / {info['max_value']:.3f}")
    print(f"Durchschnitt: {info['mean_value']:.3f} (±{info['std_value']:.3f})")
    print(f"Null-Einträge: {info['zero_entries']} ({info['zero_entries'] / q_table.size * 100:.1f}%)")
    print(f"Gelernte Einträge: {info['nonzero_entries']} ({info['nonzero_entries'] / q_table.size * 100:.1f}%)")


def display_full_q_table_matrix(q_table, scenario_name):
    """Komplette Q-Tabelle als Matrix anzeigen"""
    if q_table is None:
        return

    print(f"\n{'=' * 80}")
    print(f"KOMPLETTE Q-TABELLE MATRIX: {scenario_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Form: {q_table.shape} (States x Actions)")
    print(f"Aktionen: 0=oben, 1=rechts, 2=unten, 3=links")
    print("-" * 80)

    # Numpy Array formatiert ausgeben
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(q_table)
    np.set_printoptions()  # Zurück zu Standard-Einstellungen


def display_q_table_sample(q_table, scenario_name):
    """Ausschnitt der Q-Tabelle anzeigen"""
    if q_table is None:
        return

    print(f"\n{'=' * 60}")
    print(f"Q-TABELLE AUSSCHNITT: {scenario_name.upper()}")
    print(f"({'=' * 60}")
    print(f"Erste {DISPLAY_ROWS} Zustände, erste {DISPLAY_COLS} Aktionen:")
    print("-" * 60)

    # Header für Aktionen
    header = "State".ljust(8)
    for action in range(min(DISPLAY_COLS, q_table.shape[1])):
        header += f"Action {action}".rjust(12)
    print(header)
    print("-" * 60)

    # Q-Werte anzeigen
    for state in range(min(DISPLAY_ROWS, q_table.shape[0])):
        row = f"{state}".ljust(8)
        for action in range(min(DISPLAY_COLS, q_table.shape[1])):
            value = q_table[state, action]
            row += f"{value:10.3f}".rjust(12)
        print(row)


def display_best_actions(q_table, scenario_name):
    """Beste Aktionen für erste Zustände anzeigen"""
    if q_table is None:
        return

    print(f"\n{'=' * 60}")
    print(f"BESTE AKTIONEN: {scenario_name.upper()}")
    print(f"{'=' * 60}")

    best_actions = find_best_actions_per_state(q_table, DISPLAY_ROWS)

    print("State".ljust(8) + "Best Action".ljust(15) + "Q-Value".ljust(12))
    print("-" * 35)

    for state, action, value in best_actions:
        action_name = get_action_name(action)
        print(f"{state}".ljust(8) + f"{action} ({action_name})".ljust(15) + f"{value:.3f}".ljust(12))


def get_action_name(action):
    """Aktion zu verständlichem Namen konvertieren"""
    action_names = {
        0: "oben",
        1: "rechts",
        2: "unten",
        3: "links"
    }
    return action_names.get(action, "unbekannt")


# ============================================================================
# Hauptfunktionen
# ============================================================================

def inspect_single_q_table(scenario_name):
    """Einzelne Q-Tabelle inspizieren"""
    if scenario_name not in AVAILABLE_Q_TABLES:
        print(f"FEHLER: Unbekanntes Szenario '{scenario_name}'")
        print(f"Verfügbare Szenarien: {list(AVAILABLE_Q_TABLES.keys())}")
        return

    filepath = AVAILABLE_Q_TABLES[scenario_name]
    q_table = load_q_table_for_inspection(filepath)

    if q_table is not None:
        display_q_table_overview(q_table, scenario_name)
        display_q_table_sample(q_table, scenario_name)
        display_best_actions(q_table, scenario_name)
        display_full_q_table_matrix(q_table, scenario_name)


def inspect_all_q_tables():
    """Alle verfügbaren Q-Tabellen inspizieren"""
    print("Suche nach verfügbaren Q-Tabellen...")

    found_tables = []
    for scenario_name, filepath in AVAILABLE_Q_TABLES.items():
        if os.path.exists(filepath):
            found_tables.append(scenario_name)
        else:
            print(f"Nicht gefunden: {filepath}")

    if not found_tables:
        print("Keine Q-Tabellen gefunden!")
        return

    print(f"Gefundene Q-Tabellen: {found_tables}")

    for scenario_name in found_tables:
        inspect_single_q_table(scenario_name)


def inspect_current_scenario():
    """Q-Tabelle des aktuellen Szenarios inspizieren"""
    print(f"Inspiziere Q-Tabelle für aktuelles Szenario: {ENV_MODE}")
    inspect_single_q_table(ENV_MODE)


def compare_q_table_shapes():
    """Formen aller Q-Tabellen vergleichen"""
    print(f"\n{'=' * 60}")
    print("Q-TABELLEN FORMEN-VERGLEICH")
    print(f"{'=' * 60}")

    print("Szenario".ljust(20) + "Shape".ljust(15) + "States".ljust(10) + "Actions".ljust(10) + "Status")
    print("-" * 70)

    for scenario_name, filepath in AVAILABLE_Q_TABLES.items():
        if os.path.exists(filepath):
            q_table = load_q_table_for_inspection(filepath)
            if q_table is not None:
                shape = q_table.shape
                status = "✓ Geladen"
            else:
                shape = "N/A"
                status = "✗ Fehler"
        else:
            shape = "N/A"
            status = "✗ Nicht gefunden"

        if isinstance(shape, tuple):
            shape_str = f"{shape[0]}x{shape[1]}"
            states_str = str(shape[0])
            actions_str = str(shape[1])
        else:
            shape_str = str(shape)
            states_str = "N/A"
            actions_str = "N/A"

        print(scenario_name.ljust(20) + shape_str.ljust(15) + states_str.ljust(10) + actions_str.ljust(10) + status)


# ============================================================================
# Ausführung
# ============================================================================

def main():
    """Hauptfunktion mit Benutzerinteraktion"""
    print("Q-TABELLEN INSPEKTOR")
    print("=" * 40)
    print("1. Aktuelles Szenario inspizieren")
    print("2. Bestimmtes Szenario inspizieren")
    print("3. Alle verfügbaren Q-Tabellen inspizieren")
    print("4. Q-Tabellen-Formen vergleichen")
    print("5. Nur Matrix anzeigen (bestimmtes Szenario)")

    try:
        choice = input("\nWählen Sie eine Option (1-5): ").strip()

        if choice == "1":
            inspect_current_scenario()
        elif choice == "2":
            print(f"Verfügbare Szenarien: {list(AVAILABLE_Q_TABLES.keys())}")
            scenario = input("Szenario eingeben: ").strip()
            inspect_single_q_table(scenario)
        elif choice == "3":
            inspect_all_q_tables()
        elif choice == "4":
            compare_q_table_shapes()
        elif choice == "5":
            print(f"Verfügbare Szenarien: {list(AVAILABLE_Q_TABLES.keys())}")
            scenario = input("Szenario für Matrix-Anzeige eingeben: ").strip()
            if scenario in AVAILABLE_Q_TABLES:
                filepath = AVAILABLE_Q_TABLES[scenario]
                q_table = load_q_table_for_inspection(filepath)
                display_full_q_table_matrix(q_table, scenario)
            else:
                print(f"FEHLER: Unbekanntes Szenario '{scenario}'")
        else:
            print("Ungültige Auswahl. Zeige alle Q-Tabellen...")
            inspect_all_q_tables()

    except KeyboardInterrupt:
        print("\nProgramm beendet.")


if __name__ == "__main__":
    main()