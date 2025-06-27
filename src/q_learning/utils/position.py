# utils/position.py

from src.shared.config import GRID_SIZE

# Extraktion der Position aus Observation
def get_position(obs, env_mode):
    if env_mode == "container":
        return (obs[0], obs[1])
    return divmod(obs, GRID_SIZE)

# Konvertierung von Position zu State-Index für Grid
def pos_to_state_grid(pos, grid_size=GRID_SIZE):
    return pos[0] * grid_size + pos[1]

# Konvertierung von State-Index zu Position für Grid
def state_to_pos_grid(state, grid_size=GRID_SIZE):
    return (state // grid_size, state % grid_size)