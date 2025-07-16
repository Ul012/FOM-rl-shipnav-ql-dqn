# grid_environment.py

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
from src.shared.config import (REWARDS, GRID_SIZE, N_ACTIONS, DEFAULT_START_POS,
                               DEFAULT_GOAL_POS, DEFAULT_OBSTACLES, DEBUG_MODE, LOOP_THRESHOLD, MAX_STEPS)


# ============================================================================
# GridEnvironment Klasse
# ============================================================================

class GridEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, mode="static"):
        super(GridEnvironment, self).__init__()
        self.mode = mode
        self.grid_size = GRID_SIZE  # Aus config statt hardcoded
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(N_ACTIONS)  # Aus config statt hardcoded

        self.max_steps = MAX_STEPS
        self.loop_threshold = LOOP_THRESHOLD
        self.np_random = None

        self._initialize_environment()

    # Initialisierung der Umgebungsparameter
    def _initialize_environment(self):
        self.visited_states = {}
        self.current_steps = 0

        # Standard-Layout aus legacy_config_ql.py
        self.start_pos = DEFAULT_START_POS
        self.goal_pos = DEFAULT_GOAL_POS
        self.obstacles = DEFAULT_OBSTACLES

        self.agent_pos = self.start_pos
        self.state = self.pos_to_state(self.start_pos)

    # ============================================================================
    # Hilfsfunktionen
    # ============================================================================

    # Anpassung der Positionen basierend auf dem Modus
    def _set_positions_by_mode(self):
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

        if self.mode == "random_start":
            # Nur start_pos ändern, goal_pos und obstacles bleiben DEFAULT
            possible_starts = all_positions.copy()
            possible_starts.remove(self.goal_pos)  # DEFAULT_GOAL_POS
            for obstacle in self.obstacles:  # DEFAULT_OBSTACLES
                if obstacle in possible_starts:
                    possible_starts.remove(obstacle)

            if self.np_random is not None:
                start_idx = self.np_random.integers(0, len(possible_starts))
                self.start_pos = possible_starts[start_idx]
            else:
                import random
                self.start_pos = random.choice(possible_starts)

        elif self.mode == "random_goal":
            # Nur goal_pos ändern, start_pos und obstacles bleiben DEFAULT
            possible_goals = all_positions.copy()
            possible_goals.remove(self.start_pos)  # DEFAULT_START_POS
            for obstacle in self.obstacles:  # DEFAULT_OBSTACLES
                if obstacle in possible_goals:
                    possible_goals.remove(obstacle)

            if self.np_random is not None:
                goal_idx = self.np_random.integers(0, len(possible_goals))
                self.goal_pos = possible_goals[goal_idx]
            else:
                import random
                self.goal_pos = random.choice(possible_goals)

        elif self.mode == "random_obstacles":
            # Nur obstacles ändern, start_pos und goal_pos bleiben DEFAULT
            available_positions = all_positions.copy()
            available_positions.remove(self.start_pos)  # DEFAULT_START_POS
            available_positions.remove(self.goal_pos)   # DEFAULT_GOAL_POS

            if self.np_random is not None:
                obstacle_indices = self.np_random.choice(len(available_positions), 3, replace=False)
                self.obstacles = [available_positions[i] for i in obstacle_indices]
            else:
                import random
                self.obstacles = random.sample(available_positions, k=3)

        # static: alles bleibt bei DEFAULT-Werten

        self.agent_pos = self.start_pos
        self.state = self.pos_to_state(self.start_pos)

    # Konvertierung von Position zu State-Index
    def pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]

    # Konvertierung von State-Index zu Position
    def state_to_pos(self, state):
        return (state // self.grid_size, state % self.grid_size)

    # Berechnung der neuen Position nach Aktionsausführung
    def get_next_position(self, current_pos, action):
        row, col = current_pos
        new_row, new_col = row, col

        if action == 0 and row > 0:  # UP
            new_row = row - 1
        elif action == 1 and col < self.grid_size - 1:  # RIGHT
            new_col = col + 1
        elif action == 2 and row < self.grid_size - 1:  # DOWN
            new_row = row + 1
        elif action == 3 and col > 0:  # LEFT
            new_col = col - 1

        return (new_row, new_col)

    # Berechnung des Rewards basierend auf Terminierungsgrund
    def calculate_reward(self, next_pos, terminated_reason=None):
        if terminated_reason == "goal":
            return REWARDS["goal"]

        reward = REWARDS["step"]

        if terminated_reason == "obstacle":
            reward += REWARDS["obstacle"]
        elif terminated_reason == "loop":
            reward += REWARDS["loop_abort"]
        elif terminated_reason == "timeout":
            reward += REWARDS["timeout"]

        return reward

    # Überprüfung der Terminierungsbedingungen
    def check_termination(self, next_pos, next_state):
        terminated = False
        reason = None

        if next_pos == self.goal_pos:
            terminated = True
            reason = "goal"
        elif next_pos in self.obstacles:
            terminated = True
            reason = "obstacle"
        elif self.visited_states.get(next_state, 0) >= self.loop_threshold:
            terminated = True
            reason = "loop"
        elif self.current_steps >= self.max_steps:
            terminated = True
            reason = "timeout"

        return terminated, reason

    # Aktualisierung der besuchten Zustände für Schleifenerkennung
    def update_visited_states(self, next_state):
        if next_state in self.visited_states:
            self.visited_states[next_state] += 1
        else:
            self.visited_states[next_state] = 1

    # ============================================================================
    # Hauptfunktionen
    # ============================================================================

    # Reset der Umgebung für neue Episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._initialize_environment()
        self._set_positions_by_mode()

        self.visited_states = {}
        self.current_steps = 0

        if self.agent_pos == self.goal_pos:
            print(f"WARNING: Agent startet bereits am Ziel! Start={self.agent_pos}, Ziel={self.goal_pos}")

        if DEBUG_MODE:
            print(
                f"Grid-Umgebung reset: Modus={self.mode}, Start={self.start_pos}, Ziel={self.goal_pos}, Hindernisse={self.obstacles}, Seed={seed}")

        return self.state, {}

    # Ausführung einer Aktion in der Umgebung
    def step(self, action):
        current_pos = self.state_to_pos(self.state)
        next_pos = self.get_next_position(current_pos, action)
        next_state = self.pos_to_state(next_pos)

        self.state = next_state
        self.agent_pos = next_pos
        self.current_steps += 1

        self.update_visited_states(next_state)
        terminated, reason = self.check_termination(next_pos, next_state)
        reward = self.calculate_reward(next_pos, reason)

        return next_state, reward, terminated, False, {}

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