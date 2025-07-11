# config.py - Erweiterte Version für Q-Learning und DQN

import os
from pathlib import Path

# ============================================================================
# Basis-Parameter
# ============================================================================

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

# ============================================================================
# Umgebungskonfiguration (Q-Learning + DQN)
# ============================================================================

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
# Q-Learning Parameter
# ============================================================================

ALPHA = 0.1  # Lernrate
GAMMA = 0.95  # Diskontierungsfaktor
EPSILON = 0.1  # Explorationsrate (initial: 0.1)
EPISODES = 500  # Trainings-Episoden

# ============================================================================
# DQN Parameter
# ============================================================================

# Network Architecture
DQN_STATE_SIZE = 18  # State vector size for DQN
DQN_HIDDEN_SIZE = 128  # Hidden layer size
DQN_LEARNING_RATE = 0.001  # Learning rate for neural network

# Training Parameters
DQN_EPISODES = 500  # Training episodes
DQN_BATCH_SIZE = 32  # Batch size for experience replay
DQN_BUFFER_SIZE = 5000  # Experience replay buffer size
DQN_TARGET_UPDATE_FREQ = 100  # Target network update frequency

# Exploration Parameters
DQN_EPSILON_START = 1.0  # Initial exploration rate
DQN_EPSILON_END = 0.01  # Final exploration rate
DQN_EPSILON_DECAY = 0.995  # Exploration decay rate

# ============================================================================
# Training Parameter (Gemeinsam)
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

RESULTS_PATH = "results/"
PLOTS_PATH = "plots/"

# Dateipfad-Templates (nur Strings, keine Funktionen)
Q_TABLE_PATH_TEMPLATE = "q_table_{}.npy"  # Format: q_table_static.npy
DQN_MODEL_PATH_TEMPLATE = "dqn_model_{}.pth"  # Format: dqn_model_static.pth

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

# Spezifische Export-Pfade (selbsterklärend)
EXPORT_PATH_DQN = os.path.join(os.path.dirname(__file__), "..", "dqn", "exports")
EXPORT_PATH_QL = os.path.join(os.path.dirname(__file__), "..", "q_learning", "exports")
EXPORT_PATH_COMP = os.path.join(os.path.dirname(__file__), "..", "comparison", "exports")

# Erstelle Export-Verzeichnisse beim Import
for _path in [EXPORT_PATH_DQN, EXPORT_PATH_QL, EXPORT_PATH_COMP]:
    os.makedirs(_path, exist_ok=True)

# ============================================================================
# Debug Parameter
# ============================================================================

DEBUG_MODE = False  # Debug-Ausgaben aktivieren
VERBOSE_TRAINING = True  # Detaillierte Training-Ausgaben