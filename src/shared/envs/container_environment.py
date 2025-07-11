# container_environment.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Drittanbieter
import gymnasium as gym
from gymnasium import spaces

# Lokale Module
from src.shared.config import (REWARDS, GRID_SIZE, N_ACTIONS, CONTAINER_START_POS,
                               CONTAINER_OBSTACLES, DEBUG_MODE)


# ============================================================================
# ContainerShipEnv Klasse
# ============================================================================

class ContainerShipEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(ContainerShipEnv, self).__init__()
        self.grid_size = GRID_SIZE  # Aus config statt hardcoded
        self.start_pos = CONTAINER_START_POS  # Aus config
        self.obstacles = CONTAINER_OBSTACLES  # Aus config
        self.max_steps = 300

        self.observation_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, 2])
        self.action_space = spaces.Discrete(N_ACTIONS)  # Aus config statt hardcoded

        self.np_random = None
        self._initialize_environment()

    # Initialisierung der Umgebungsparameter
    def _initialize_environment(self):
        self.agent_pos = self.start_pos
        self.container_loaded = False
        self.steps = 0
        self.visited_states = {}
        self.max_loop_count = 3
        self.successful_dropoffs = 0

    # ============================================================================
    # Hilfsfunktionen
    # ============================================================================

    # Generierung zufälliger Pickup- und Dropoff-Positionen
    def _set_random_positions(self):
        positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        positions.remove(self.start_pos)

        for obstacle in self.obstacles:
            if obstacle in positions:
                positions.remove(obstacle)

        if self.np_random is not None:
            pickup_idx = self.np_random.integers(0, len(positions))
            self.pickup_pos = positions.pop(pickup_idx)

            dropoff_idx = self.np_random.integers(0, len(positions))
            self.dropoff_pos = positions[dropoff_idx]
        else:
            import random
            self.pickup_pos = random.choice(positions)
            positions.remove(self.pickup_pos)
            self.dropoff_pos = random.choice(positions)

    # Erstellung der aktuellen Beobachtung
    def _get_obs(self):
        return (self.agent_pos[0], self.agent_pos[1], int(self.container_loaded))

    # Konvertierung von Position zu State-Index
    def pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]

    # Berechnung der neuen Position nach Aktionsausführung
    def get_next_position(self, current_pos, action):
        x, y = current_pos

        if action == 0 and x > 0:  # UP
            x -= 1
        elif action == 1 and y < self.grid_size - 1:  # RIGHT
            y += 1
        elif action == 2 and x < self.grid_size - 1:  # DOWN
            x += 1
        elif action == 3 and y > 0:  # LEFT
            y -= 1

        return (x, y)

    # Berechnung des Rewards basierend auf Terminierungsgrund
    # NEUE VERSION - KONSISTENT MIT GRIDENVIRONMENT:
    def calculate_reward(self, terminated_reason=None):
        """
        Berechnet Rewards konsistent mit GridEnvironment.
        Basis-Schritt-Reward + spezifische Belohnungen/Strafen.
        """
        # Basis-Reward für jeden Schritt
        base_reward = REWARDS["step"]  # -1

        if terminated_reason == "dropoff":
            # Positive Belohnung ZUSÄTZLICH zum Schritt
            return base_reward + REWARDS["dropoff"]  # -1 + 20 = 19
        elif terminated_reason == "pickup":
            # Positive Belohnung ZUSÄTZLICH zum Schritt
            return base_reward + REWARDS["pickup"]  # -1 + 8 = 7
        elif terminated_reason == "obstacle":
            # Negative Strafe ZUSÄTZLICH zum Schritt
            return base_reward + REWARDS["obstacle"]  # -1 + (-10) = -11
        elif terminated_reason == "loop":
            # Negative Strafe ZUSÄTZLICH zum Schritt
            return base_reward + REWARDS["loop_abort"]  # -1 + (-10) = -11
        elif terminated_reason == "timeout":
            # Negative Strafe ZUSÄTZLICH zum Schritt
            return base_reward + REWARDS["timeout"]  # -1 + (-10) = -11
        else:
            # Nur Basis-Schritt-Reward
            return base_reward  # -1

    # Überprüfung der Terminierungsbedingungen
    def check_termination_and_rewards(self, next_pos, state_key):
        terminated = False
        reason = None

        if self.visited_states.get(state_key, 0) >= self.max_loop_count:
            terminated = True
            reason = "loop"
        elif self.steps >= self.max_steps:
            terminated = True
            reason = "timeout"
        elif next_pos in self.obstacles:
            terminated = True
            reason = "obstacle"
        elif not self.container_loaded and next_pos == self.pickup_pos:
            self.container_loaded = True
            reason = "pickup"
        elif self.container_loaded and next_pos == self.dropoff_pos:
            terminated = True
            reason = "dropoff"
            self.successful_dropoffs += 1

        return terminated, reason

    # Aktualisierung der besuchten Zustände für Schleifenerkennung
    def update_visited_states(self, state_key):
        if state_key in self.visited_states:
            self.visited_states[state_key] += 1
        else:
            self.visited_states[state_key] = 1

    # ============================================================================
    # Hauptfunktionen
    # ============================================================================

    # Reset der Umgebung für neue Episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_environment()
        self._set_random_positions()

        obs = self._get_obs()

        if DEBUG_MODE:
            print(
                f"Container-Umgebung reset: Start={self.start_pos}, Pickup={self.pickup_pos}, Dropoff={self.dropoff_pos}, Seed={seed}")

        return obs, {}

    # Ausführung einer Aktion in der Umgebung
    def step(self, action):
        next_pos = self.get_next_position(self.agent_pos, action)
        self.agent_pos = next_pos
        self.steps += 1

        obs = self._get_obs()
        state_key = (obs[0], obs[1], obs[2])

        self.update_visited_states(state_key)
        terminated, reason = self.check_termination_and_rewards(next_pos, state_key)
        reward = self.calculate_reward(reason)

        return obs, reward, terminated, False, {}

    # ============================================================================
    # Zusätzliche Methoden für Kompatibilität
    # ============================================================================

    # Seed-Konfiguration für ältere Gym-Versionen
    def seed(self, seed=None):
        from gymnasium.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Seed-Konfiguration für Action Space
    def seed_action_space(self, seed=None):
        self.action_space.seed(seed)
        return seed