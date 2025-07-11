# utils/qlearning.py

import numpy as np
from src.shared.config import QL_ALPHA, GAMMA, ENV_MODE, N_ACTIONS

# Initialisierung der Q-Tabelle
def initialize_q_table(env):
    n_states = env.observation_space.n if hasattr(env.observation_space, 'n') else np.prod(env.observation_space.nvec)
    n_actions = N_ACTIONS
    Q = np.zeros((n_states, n_actions))
    print(f"Q-Tabelle initialisiert: {n_states} Zustände, {n_actions} Aktionen")
    return Q, n_states, n_actions

# Epsilon-greedy Aktionsauswahl
def select_action(Q, state, epsilon, n_actions=N_ACTIONS):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

# Q-Wert Update (Q-Learning)
def update_q_value(Q, state, action, reward, next_state, alpha=QL_ALPHA, gamma=GAMMA):
    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

# Speicherung der Q-Tabelle
def save_q_table(Q, env_mode=ENV_MODE):
    filepath = f"q_table_{env_mode}.npy"
    np.save(filepath, Q)
    print(f"Q-Tabelle gespeichert: {filepath}")

# Laden der Q-Tabelle
def load_q_table(filepath):
    try:
        Q = np.load(filepath)
        print(f"Q-Tabelle geladen: {filepath}")
        return Q
    except FileNotFoundError:
        print(f"FEHLER: Q-Tabelle nicht gefunden: {filepath}")
        return None

# Bestimmung der optimalen Aktion für einen Zustand
def get_best_action(Q, state):
    return np.argmax(Q[state])