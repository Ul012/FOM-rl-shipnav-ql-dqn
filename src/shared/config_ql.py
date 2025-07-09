# config_ql.py

# ============================================================================
# Basis-Parameter
# ============================================================================

# Grid Constants
GRID_SIZE = 5

# Aktionen (Richtungen)
ACTIONS = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3
}
N_ACTIONS = len(ACTIONS)

# Standard Grid Layout (für alle Grid-Modi)
DEFAULT_START_POS = (0, 0)
DEFAULT_GOAL_POS = (4, 4)
DEFAULT_OBSTACLES = [(1, 1), (2, 3), (3, 1)]

# Container Environment Layout
CONTAINER_START_POS = (0, 0)
CONTAINER_OBSTACLES = [(1, 3), (1, 2), (3, 1)]

# ============================================================================
# Umgebungskonfiguration
# ============================================================================

ENV_MODE = "static"  # Optionen: static, random_start, random_goal, random_obstacles, container
SEED = 42  # Random seed für Reproduzierbarkeit

# ============================================================================
# Rewardsystem
# ============================================================================

REWARDS = {
    "step": -1,
    "goal": 10,
    "obstacle": -10,
    "loop_abort": -10,
    "timeout": -10,
    "pickup": 8,
    "dropoff": 20
}

# ============================================================================
# Q-Learning Parameter
# ============================================================================

ALPHA = 0.1  # Lernrate
GAMMA = 0.95  # Diskontierungsfaktor
EPSILON = 0.1  # Explorationsrate (initial: 0.1)
EPISODES = 500  # Trainings-Episoden

# ============================================================================
# Training Parameter
# ============================================================================

MAX_STEPS = 100  # Max. Schritte pro Episode
LOOP_THRESHOLD = 10  # Schleifenwiederholungen für Abbruch (initial: 6)

# ============================================================================
# Evaluation Parameter
# ============================================================================

EVAL_EPISODES = 500  # Anzahl Episoden für Evaluation
EVAL_MAX_STEPS = 50  # Max. Schritte pro Episode in Evaluation (initial: 100)

# ============================================================================
# Dateipfade
# ============================================================================

# Q-Tabelle Pfad basierend auf Modus
def get_q_table_path(env_mode):
    return f"q_table_{env_mode}.npy"

RESULTS_PATH = "results/"
PLOTS_PATH = "plots/"

# ============================================================================
# Visualisierung Parameter
# ============================================================================

SHOW_VALUES = True  # Q-Werte in Heatmaps anzeigen
COLORMAP_STYLE = "viridis"  # Colormap für Heatmaps
VISUALIZATION_DELAY = 0.5  # Verzögerung zwischen Schritten (Sekunden)

CELL_SIZE = 80  # Größe einer Grid-Zelle in Pixeln
FRAME_DELAY = 0.4  # Verzögerung zwischen Frames (Sekunden)
ARROW_SCALE = 0.3  # Größe der Policy-Pfeile
SHOW_GRID_LINES = True  # Grid-Linien in Visualisierungen

FIGURE_SIZE = (10, 6)  # Plot-Größe
DPI_SETTING = 100  # Auflösung für gespeicherte Plots

# ============================================================================
# Export Parameter
# ============================================================================

EXPORT_PDF = True  # PDF-Export für Visualisierungen
EXPORT_PATH = "../dqn/exports/"  # Pfad für exportierte Dateien

# ============================================================================
# Debug Parameter
# ============================================================================

DEBUG_MODE = False  # Debug-Ausgaben aktivieren
VERBOSE_TRAINING = True  # Detaillierte Training-Ausgaben