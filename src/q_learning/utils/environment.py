# utils/environment.py

import sys
import os

# Project root setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Imports at top level
from src.shared.envs.grid_environment import GridEnvironment
from src.shared.envs.container_environment import ContainerShipEnv


def initialize_environment(env_mode):
    """Initialize environment based on mode."""
    env = ContainerShipEnv() if env_mode == "container" else GridEnvironment(mode=env_mode)
    grid_size = env.grid_size
    print(f"Umgebung initialisiert: {env_mode}-Modus, Grid-Größe: {grid_size}x{grid_size}")
    return env, grid_size


def initialize_environment_for_scenario(scenario_config):
    """Initialize environment for scenario comparison."""
    if scenario_config["environment"] == "container":
        env = ContainerShipEnv()
    else:
        env = GridEnvironment(mode=scenario_config["env_mode"])
    return env, env.grid_size