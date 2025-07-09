# 🚢 Q-Learning vs Deep Q-Learning für Schiffsnavigation

Vergleichsprojekt zwischen Q-Learning und Deep Q-Learning (DQN) Algorithmen für die Navigation eines Schiffs in einem 5x5 Gitter.

## 🎯 Projektziele

- Implementierung von Q-Learning und DQN für Navigationsprobleme
- Vergleich beider Algorithmen unter identischen Bedingungen
- Evaluation in 5 verschiedenen Umgebungsszenarien
- Bereitstellung reproduzierbarer Experimente

## 🧠 Algorithmen

### Q-Learning
- Tabellenbasierte Q-Funktion
- Epsilon-Greedy Exploration
- Für diskrete Zustandsräume

### Deep Q-Learning (DQN)
- Neural Network Q-Funktion
- Experience Replay
- Target Network Updates

## 📁 Projektstruktur

```
ship-navigation-ql-dqn/
├── src/
│   ├── q_learning/          # Q-Learning Implementation
│   ├── dqn/                 # Deep Q-Learning Implementation
│   ├── comparison/          # Algorithmus-Vergleich
│   └── shared/              # Gemeinsame Komponenten
│       ├── config.py        # Zentrale Konfiguration
│       └── envs/            # Environment-Implementierungen
├── docs/                    # Dokumentation
└── exports/                 # Ergebnisse
```

## 🗺️ Szenarien

| Szenario | ENV_MODE | Beschreibung |
|----------|----------|--------------|
| **Static** | `"static"` | Feste Positionen |
| **Random Start** | `"random_start"` | Variable Startposition |
| **Random Goal** | `"random_goal"` | Variable Zielposition |
| **Random Obstacles** | `"random_obstacles"` | Variable Hindernisse |
| **Container** | `"container"` | Pickup/Dropoff-Aufgabe |

## ⚙️ Technische Spezifikationen

- **Umgebung**: 5x5 Grid (OpenAI Gymnasium)
- **Zustandsraum**: 25 Zustände (Grid), 50 Zustände (Container)
- **Aktionsraum**: 4 Richtungen (↑→↓←)
- **Implementierung**: Python mit PyTorch (DQN)

## 🚀 Schnellstart

```bash
# Installation
git clone https://github.com/Ul012/FOM-rl-shipnav-ql-dql.git
cd FOM-rl-shipnav-ql-dql
pip install -r requirements.txt

# Q-Learning Training
cd src/q_learning
python train.py

# DQN Training  
cd src/dqn
python train.py --mode static --episodes 500

# Algorithmus-Vergleich
cd src/comparison
python compare_algorithms.py
```

---

**📚 Weitere Informationen:**
- [Setup](setup.md) - Installation und Konfiguration
- [Training](training.md) - Verwendung beider Algorithmen
- [Funktionsweise](funktionsweise.md) - Technische Details