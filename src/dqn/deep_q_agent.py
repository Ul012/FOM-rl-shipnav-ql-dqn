# src/dqn/deep_q_agent.py

import random
import sys
import os
from collections import deque, namedtuple
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.shared.config import ACTIONS, GRID_SIZE, REWARDS
from src.shared.config_utils import get_dqn_config

# Experience tuple für Replay Buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """Deep Q-Network für die Schiffsnavigation mit gymnasium Environments"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DeepQLearningAgent:
    """Deep Q-Learning Agent für gymnasium Environments"""

    def __init__(self, **kwargs):
        # Lade Konfiguration
        config = get_dqn_config()
        config.update(kwargs)  # Überschreibe mit übergebenen Parametern

        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.learning_rate = config['learning_rate']
        self.discount_factor = config['discount_factor']
        self.exploration_rate = config['exploration_rate']
        self.exploration_decay = config['exploration_decay']
        self.min_exploration_rate = config['min_exploration_rate']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        self.buffer_size = config['buffer_size']
        self.hidden_size = config['hidden_size']
        self.seed = config.get('seed', None)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Seed für Reproduzierbarkeit
        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Netzwerke
        self.q_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Experience Replay
        self.memory = deque(maxlen=self.buffer_size)
        self.update_counter = 0

        # Training metrics
        self.losses = []
        self.training_metrics = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'total_steps': 0,
            'total_reward': 0.0
        }

        # Initialisiere target network
        self._update_target_network()

    def _get_state_representation(self, observation, env_type="grid") -> np.ndarray:
        """
        Erstellt eine numerische State-Repräsentation für verschiedene Environment-Typen.

        Args:
            observation: Beobachtung vom Environment
            env_type: "grid" für GridEnvironment, "container" für ContainerShipEnv
        """
        if env_type == "grid":
            # GridEnvironment: observation ist state index
            state_idx = observation
            row, col = state_idx // GRID_SIZE, state_idx % GRID_SIZE

            # Normalisierte Position
            pos_x = col / (GRID_SIZE - 1)
            pos_y = row / (GRID_SIZE - 1)

            # One-hot encoding der Position
            position_onehot = np.zeros(GRID_SIZE * GRID_SIZE)
            position_onehot[state_idx] = 1.0

            # Kombiniere Features
            state = np.array([pos_x, pos_y] + position_onehot.tolist(), dtype=np.float32)

        elif env_type == "container":
            # ContainerShipEnv: observation ist (x, y, container_loaded)
            x, y, container_loaded = observation

            # Normalisierte Position
            pos_x = x / (GRID_SIZE - 1)
            pos_y = y / (GRID_SIZE - 1)

            # Container status
            container_status = float(container_loaded)

            # One-hot encoding der Position
            position_onehot = np.zeros(GRID_SIZE * GRID_SIZE)
            state_idx = x * GRID_SIZE + y
            position_onehot[state_idx] = 1.0

            # Kombiniere Features
            state = np.array([pos_x, pos_y, container_status] + position_onehot.tolist(), dtype=np.float32)

        else:
            raise ValueError(f"Unbekannter Environment-Typ: {env_type}")

        # Padding falls benötigt, um auf state_size zu kommen
        if len(state) < self.state_size:
            padding = np.zeros(self.state_size - len(state))
            state = np.concatenate([state, padding])
        elif len(state) > self.state_size:
            state = state[:self.state_size]

        return state

    def select_action(self, observation, valid_actions=None, training=True, env_type="grid"):
        """
        Wählt eine Aktion basierend auf der Epsilon-Greedy-Strategie.

        Args:
            observation: Beobachtung vom Environment
            valid_actions: Liste gültiger Aktionen (optional)
            training: Ob Training-Modus (Exploration) oder Evaluation
            env_type: Environment-Typ für State-Repräsentation
        """
        state = self._get_state_representation(observation, env_type)

        # Standardmäßig alle Aktionen erlaubt
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if not valid_actions:
            return 0  # Fallback

        # Epsilon-Greedy für Training
        if training and random.random() < self.exploration_rate:
            action = random.choice(valid_actions)
        else:
            # Greedy action selection
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.q_network(state_tensor)

            # Maskiere ungültige Aktionen
            masked_q_values = q_values.clone()
            for i in range(self.action_size):
                if i not in valid_actions:
                    masked_q_values[0][i] = float('-inf')

            action = masked_q_values.argmax().item()

        return action

    def update(self, state, action, reward, next_state, done, env_type="grid"):
        """
        Speichert Erfahrung und trainiert das Netzwerk.

        Args:
            state: Aktueller Zustand
            action: Ausgeführte Aktion
            reward: Erhaltene Belohnung
            next_state: Nächster Zustand
            done: Episode beendet
            env_type: Environment-Typ
        """
        # Konvertiere Zustände zu State-Repräsentationen
        state_repr = self._get_state_representation(state, env_type)
        next_state_repr = self._get_state_representation(next_state, env_type)

        # Speichere Erfahrung
        experience = Experience(state_repr, action, reward, next_state_repr, done)
        self.memory.append(experience)

        # Trainiere, wenn genug Erfahrungen vorhanden
        if len(self.memory) >= self.batch_size:
            self._replay_experience()

    def _replay_experience(self):
        """Trainiert das Netzwerk mit einem Batch von Erfahrungen."""
        batch = random.sample(self.memory, self.batch_size)

        # Konvertiere zu numpy arrays, dann zu Tensoren
        states = torch.FloatTensor(np.stack([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.stack([e.next_state for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e.done for e in batch])).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Nächste Q-Werte des Target Networks
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)

        # Berechne Verlust
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimiere
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update Target Network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()

        # Merke Verlust
        self.losses.append(loss.item())

    def _update_target_network(self):
        """Kopiert Gewichte vom Hauptnetzwerk zum Zielnetzwerk."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_exploration(self):
        """Reduziert die Explorationsrate."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

    def episode_finished(self, success: bool, steps: int, total_reward: float):
        """Wird am Ende jeder Episode aufgerufen."""
        self.training_metrics['total_episodes'] += 1
        if success:
            self.training_metrics['successful_episodes'] += 1
        self.training_metrics['total_steps'] += steps
        self.training_metrics['total_reward'] += total_reward

    def get_metrics(self) -> Dict[str, Any]:
        """Gibt alle verfügbaren Metriken zurück."""
        recent_losses = self.losses[-100:] if self.losses else [0]

        base_metrics = {
            'exploration_rate': self.exploration_rate,
            'memory_size': len(self.memory),
            'average_loss': np.mean(recent_losses),
            'update_counter': self.update_counter,
            'device': str(self.device),
            'network_parameters': sum(p.numel() for p in self.q_network.parameters()),
            'total_episodes': self.training_metrics['total_episodes'],
            'successful_episodes': self.training_metrics['successful_episodes']
        }

        if self.training_metrics['total_episodes'] > 0:
            base_metrics.update({
                'success_rate': (self.training_metrics['successful_episodes'] /
                                 self.training_metrics['total_episodes']) * 100,
                'average_steps': self.training_metrics['total_steps'] / self.training_metrics['total_episodes'],
                'average_reward': self.training_metrics['total_reward'] / self.training_metrics['total_episodes']
            })

        return base_metrics

    def save_model(self, filepath: str):
        """Speichert das trainierte Modell."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'metrics': self.training_metrics,
            'losses': self.losses,
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_size': self.hidden_size
            }
        }, filepath)
        print(f"DQN Modell gespeichert: {filepath}")

    def load_model(self, filepath: str):
        """Lädt ein trainiertes Modell."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.exploration_rate = checkpoint.get('exploration_rate', self.exploration_rate)
            self.training_metrics = checkpoint.get('metrics', self.training_metrics)
            self.losses = checkpoint.get('losses', [])
            print(f"DQN Modell geladen: {filepath}")
        except FileNotFoundError:
            print(f"Modell-Datei nicht gefunden: {filepath}")

    def print_stats(self):
        """Gibt Statistiken aus."""
        metrics = self.get_metrics()
        print(f"\nDeep Q-Learning Agent Statistiken:")
        print(f"Exploration Rate: {metrics['exploration_rate']:.3f}")
        print(f"Memory Size: {metrics['memory_size']}")
        print(f"Average Loss: {metrics['average_loss']:.3f}")
        print(f"Device: {metrics['device']}")
        print(f"Network Parameters: {metrics['network_parameters']}")

        if 'success_rate' in metrics:
            print(f"Erfolgsrate: {metrics['success_rate']:.1f}%")
            print(f"Durchschnittliche Schritte: {metrics['average_steps']:.1f}")
            print(f"Durchschnittliche Belohnung: {metrics['average_reward']:.2f}")
            print(f"Episoden: {metrics['total_episodes']}")

    @property
    def exploration_rate_value(self) -> float:
        """Gibt die aktuelle Exploration Rate zurück."""
        return self.exploration_rate