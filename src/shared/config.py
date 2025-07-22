# config.py - Erweiterte Version für Q-Learning und DQN

import os
from pathlib import Path

# ============================================================================
# Basis-Parameter
# ============================================================================

# Hyperparameter-Setup
SETUP_NAME = "v1"  # Bei Auswahl "v2" USE_EPSILON_DECAY = True setzen. Bei "v1" auf USE_EPSILON_DECAY = False setzen. Wird für Exportverzeichnisse verwendet

# Grid Constants
GRID_SIZE = 5
SEED = 42 # Reproduzierbarkeit

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

# Umgebungskonfiguration - für beide identisch
ENV_MODE = "static"  # Optionen: static, random_start, random_goal, random_obstacles, container

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
# Training Parameter (Gemeinsam)
# ============================================================================

GAMMA = 0.95  # Diskontierungsfaktor. Mögliche Alternative: GAMMA = 0.99 # stärkerer Fokus auf Langzeit
EPISODES = 500  # Trainings-Episoden
MAX_STEPS = 100  # Max. Schritte pro Episode
LOOP_THRESHOLD = 3  # Schleifenwiederholungen für Abbruch

# Exploration Parameter
EPSILON_START = 1.0  # Initial exploration rate - beide Algorithmen
EPSILON_END = 0.01  # Final exploration rate - beide Algorithmen
EPSILON_DECAY = 0.995  # Exploration decay rate - beide Algorithmen
USE_EPSILON_DECAY = False  # v1_ Standard, festes Epsilon, v2: Epsilon Decay aktivieren

# Feste Epsilon-Werte (wenn USE_EPSILON_DECAY=False)
QL_EPSILON_FIXED = 0.1  # Q-Learning festes Epsilon
DQN_EPSILON_FIXED = 0.1  # DQN festes Epsilon (gleicher Wert für Fairness!)

# Evaluation Parameter - für beide identisch
EVAL_EPISODES = 500  # Anzahl Episoden für Evaluation - beide
EVAL_MAX_STEPS = 50  # Max. Schritte pro Episode in Evaluation - beide

# ============================================================================
# Q-LEARNING PARAMETER
# ============================================================================

# Q-Learning Hyperparameter
QL_ALPHA = 0.1  # Lernrate (learning rate) - nur Q-Learning. Mögliche Alternative: QL_ALPHA = 0.05 # geringere Lernrate

# ============================================================================
# DQN Parameter
# ============================================================================

# Network Architecture
DQN_STATE_SIZE = 25  # State vector size for DQN
DQN_HIDDEN_SIZE = 32  # Hidden layer size
DQN_LEARNING_RATE = 0.001  # Learning rate for neural network

# Training Parameters
DQN_BATCH_SIZE = 32  # Batch size for experience replay
DQN_BUFFER_SIZE = 3000  # Experience replay buffer size
DQN_TARGET_UPDATE_FREQ = 200  # Target network update frequency

# DQN Exploration Parameter - nutzt gemeinsame Epsilon-Parameter
# Verwendet abhängig von USE_EPSILON_DECAY:
# - False: DQN_EPSILON_FIXED (konstant)
# - True: EPSILON_START → EPSILON_END mit EPSILON_DECAY

# ============================================================================
# DATEIPFADE UND VERZEICHNISSE
# ============================================================================

# Ergebnisse & Plots
RESULTS_PATH = "results/"
PLOTS_PATH = "plots/"

# Dateipfad-Templates (nur Strings, keine Funktionen)
Q_TABLE_PATH_TEMPLATE = "q_table_{}.npy"  # Format: q_table_static.npy
DQN_MODEL_PATH_TEMPLATE = "dqn_model_{}.pth"  # Format: dqn_model_static.pth

# Export-Pfade
EXPORT_PATH_QL = "exports"
EXPORT_PATH_DQN = "exports"
# EXPORT_PATH_QL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "q_learning", "exports", SETUP_NAME)
# EXPORT_PATH_DQN = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dqn", "exports", SETUP_NAME)
EXPORT_PATH_COMP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "comparison", "exports")

# Erstelle Export-Verzeichnisse beim Import
for _path in [EXPORT_PATH_DQN, EXPORT_PATH_QL, EXPORT_PATH_COMP]:
    os.makedirs(_path, exist_ok=True)

# ============================================================================
# VISUALISIERUNG PARAMETER (für beide Algorithmen)
# ============================================================================

# Gemeinsame Visualisierung Parameter
SHOW_VALUES = True  # Q-Werte in Heatmaps anzeigen
COLORMAP_STYLE = "viridis"  # Colormap für Heatmaps
VISUALIZATION_DELAY = 0.5  # Verzögerung zwischen Schritten (Sekunden)
CELL_SIZE = 80  # Größe einer Grid-Zelle in Pixeln
FRAME_DELAY = 0.4  # Verzögerung zwischen Frames (Sekunden)
ARROW_SCALE = 0.3  # Größe der Policy-Pfeile
SHOW_GRID_LINES = True  # Grid-Linien in Visualisierungen
FIGURE_SIZE = (10, 6)  # Plot-Größe
DPI_SETTING = 100  # Auflösung für gespeicherte Plots
EXPORT_PDF = True  # PDF-Export für Visualisierungen

# ============================================================================
# DEBUG PARAMETER (für beide Algorithmen)
# ============================================================================

DEBUG_MODE = False  # Debug-Ausgaben aktivieren
VERBOSE_TRAINING = True  # Detaillierte Training-Ausgaben