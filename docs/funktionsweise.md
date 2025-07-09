# Funktionsweise der Algorithmen

## Q-Learning

### Algorithmus
```
Q(s,a) ← Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
```

- `α` = Lernrate (ALPHA)
- `γ` = Diskontfaktor (GAMMA)  
- `r` = Reward
- `s'` = Folgezustand

### Entscheidung
**Epsilon-Greedy-Strategie:**
- Exploration (ε): Zufällige Aktion
- Exploitation (1-ε): Beste Aktion aus Q-Tabelle

### Zustandsrepräsentation
- **Grid**: 25 Zustände (state = row × 5 + col)
- **Container**: 50 Zustände (Position + Container-Status)

## Deep Q-Learning (DQN)

### Komponenten
1. **Q-Network**: Neural Network für Q-Funktion
2. **Target Network**: Stabile Q-Targets
3. **Experience Replay**: Batch-Training aus Memory

### Netzwerk-Architektur
```python
Input: State-Vektor (18 Dimensionen)
Hidden: 128 → 128 → 128 (ReLU + Dropout)
Output: 4 Q-Werte (für 4 Aktionen)
```

### Training
```python
target_q = reward + γ × max(target_network(next_state))
loss = MSE(q_network(state)[action], target_q)
```

## Algorithmus-Vergleich

| Aspekt | Q-Learning | DQN |
|--------|------------|-----|
| **Q-Funktion** | Explizite Tabelle | Neural Network |
| **Speicher** | O(\|S\| × \|A\|) | O(Netzwerk-Parameter) |
| **Updates** | Online | Batch mit Replay |
| **Konvergenz** | Garantiert | Approximativ |

## Gemeinsame Komponenten

### Environments
Beide Algorithmen nutzen identische Gymnasium-Environments:
- **GridEnvironment**: Standard-Navigation
- **ContainerShipEnv**: Pickup/Dropoff-Aufgaben

### Reward-System
```python
REWARDS = {
    "step": -1,           # Bewegungskosten
    "goal": 10,           # Ziel erreicht
    "obstacle": -10,      # Hindernis-Kollision
    "loop_abort": -10,    # Schleifenabbruch
    "timeout": -10,       # Episode-Timeout
    "pickup": 8,          # Container aufgenommen
    "dropoff": 20         # Container abgeliefert
}
```

### Terminierung
1. **Erfolg**: Ziel erreicht / Container abgeliefert
2. **Schleife**: Zustand > LOOP_THRESHOLD mal besucht
3. **Kollision**: Hindernis getroffen
4. **Timeout**: MAX_STEPS erreicht

## Szenarien

| Modus | ENV_MODE | Beschreibung |
|-------|----------|--------------|
| **Static** | `"static"` | Feste Positionen |
| **Random Start** | `"random_start"` | Variable Startposition |
| **Random Goal** | `"random_goal"` | Variable Zielposition |
| **Random Obstacles** | `"random_obstacles"` | Variable Hindernisse |
| **Container** | `"container"` | Pickup/Dropoff mit erweitertem Zustandsraum |

## Modell-Verwaltung

### Q-Learning
```
src/q_learning/exports/
├── q_table_static.npy
├── q_table_random_start.npy
├── q_table_random_goal.npy
├── q_table_random_obstacles.npy
└── q_table_container.npy
```

### DQN
```
src/dqn/exports/
├── dqn_model_static.pth
├── dqn_model_random_start.pth
├── dqn_model_random_goal.pth
├── dqn_model_random_obstacles.pth
└── dqn_model_container.pth
```

## Reproduzierbarkeit

### Seed-Management
```python
SEED = 42  # In shared/config.py

# Beide Algorithmen verwenden:
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  # Für DQN
```

### Deterministische Faktoren
- Umgebungs-Randomisierung
- Agent-Entscheidungen
- Modell-Initialisierung

Alle Ergebnisse sind bei gleichem Seed reproduzierbar.