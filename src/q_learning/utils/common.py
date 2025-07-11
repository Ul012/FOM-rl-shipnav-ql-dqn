# utils/common.py

import random
import numpy as np
from pathlib import Path
from src.shared.config import SEED, EXPORT_PDF, EXPORT_PATH_QL, GRID_SIZE


# Seed-Konfiguration für Reproduzierbarkeit
def set_all_seeds(seed=None):
    if seed is None:
        seed = SEED

    random.seed(seed)
    np.random.seed(seed)
    print(f"Seeds gesetzt auf: {seed}")
    return seed


# Zustandscodierung je nach Umgebungstyp
def obs_to_state(obs, env_mode, grid_size=None):
    if env_mode == "container":
        if grid_size is None:
            grid_size = GRID_SIZE
        return obs[0] * grid_size + obs[1] + (grid_size * grid_size) * obs[2]
    return obs


# Erfolgserkennung je nach Umgebungstyp
def check_success(reward, env_mode):
    from src.shared.config import REWARDS
    if env_mode == "container":
        return reward == REWARDS["dropoff"]
    else:  # Grid-Environment
        return reward == REWARDS["goal"]


# Erstellung des Export-Ordners
def setup_export():
    if EXPORT_PDF:
        Path(EXPORT_PATH_QL).mkdir(exist_ok=True)