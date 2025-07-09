# RL Ship Navigation: Q-Learning vs Deep Q-Learning

Ein Vergleich zwischen Q-Learning und Deep Q-Learning Algorithmen für autonome Schiffsnavigation in Grid-Umgebungen.

## 📋 Projektübersicht

Dieses Projekt vergleicht die Leistung von klassischem Q-Learning und Deep Q-Learning (DQN) in verschiedenen Navigationsszenarien:

- **Statische Umgebung**: Feste Start-, Ziel- und Hindernis-Positionen
- **Dynamische Szenarien**: Zufällige Start-, Ziel- oder Hindernis-Positionen  
- **Container-Umgebung**: Pickup/Dropoff-Aufgaben

## 🧠 Algorithmen

### Q-Learning
- Tabellenbasierter Ansatz mit expliziter Q-Tabelle
- Epsilon-Greedy Exploration
- Optimal für kleine, diskrete Zustandsräume

### Deep Q-Learning (DQN)
- Neuronale Netzwerke approximieren Q-Funktion
- Experience Replay für stabiles Training
- Target Network für stabilere Updates
- Skaliert auf größere Zustandsräume

## 🏗️ Projektstruktur

```
ship-navigation-ql-dqn/
├── src/
│   ├── shared/           # Gemeinsame Komponenten
│   │   ├── config.py     # Gemeinsame Konfiguration
│   │   └── envs/         # Environment-Implementierungen
│   │       ├── grid_environment.py
│   │       └── container_environment.py
│   ├── q_learning/       # Q-Learning Implementation
│   │   ├── train.py      # Training
│   │   ├── evaluate_policy.py
│   │   ├── visualize_policy.py
│   │   ├── compare_scenarios.py
│   │   └── utils/        # Q-Learning spezifische Utils
│   ├── dqn/             # Deep Q-Learning Implementation
│   │   ├── deep_q_agent.py
│   │   ├── train.py
│   │   └── train_all_scenarios.py
│   ├── comparison/       # Algorithmus-Vergleich
│   │   └── compare_algorithms.py
│   └── experiments/      # Vergleichsexperimente
├── exports/             # Trainings-Ergebnisse und Plots
├── docs/               # Dokumentation
└── README.md
```

## 🚀 Installation & Setup

### Voraussetzungen
- Python 3.10+
- Git

### Installation

1. **Repository klonen:**
```bash
git clone https://github.com/Ul012/FOM-rl-shipnav-ql-dql.git
cd FOM-rl-shipnav-ql-dql
```

2. **Virtual Environment erstellen:**
```bash
python -m venv C:\venvs\ql-dqn-venv  # Windows
# oder 
python -m venv venv  # Linux/Mac

# Aktivieren:
C:\venvs\ql-dqn-venv\Scripts\activate  # Windows
# oder
source venv/bin/activate  # Linux/Mac
```

3. **Dependencies installieren:**
```bash
pip install -r requirements.txt
```

## 🎮 Verwendung

### Q-Learning Training

```bash
cd src/q_learning

# Einzelnes Szenario trainieren
python train.py

# Alle Szenarien trainieren
python train_all_scenarios.py

# Policy evaluieren
python evaluate_policy.py

# Policy visualisieren
python visualize_policy.py

# Szenarien vergleichen
python compare_scenarios.py
```

### Deep Q-Learning Training

```bash
cd src/dqn

# Einzelnes Szenario trainieren
python train.py --mode static --episodes 500

# Alle Szenarien trainieren
python train_all_scenarios.py --episodes 500 --runs 3

# Nur Evaluation (lädt gespeichertes Modell)
python train.py --mode static --eval-only
```

### Algorithmus-Vergleich

```bash
cd src/comparison

# Vollständiger Vergleich beider Algorithmen
python compare_algorithms.py --runs 5

# Schneller Test-Vergleich
python compare_algorithms.py --ql-episodes 100 --dqn-episodes 100 --runs 2
```

### Konfiguration anpassen

Editiere `src/shared/config.py` für:
- **Q-Learning**: Lernrate, Epsilon, Gamma
- **DQN**: Network-Architektur, Batch-Size, Experience Replay
- **Environment**: Grid-Größe, Rewards, Max-Steps
- **Training**: Episoden, Seeds für Reproduzierbarkeit

## 📊 Szenarien

### 1. Statisches Grid (`static`)
- Feste Positionen für Start, Ziel und Hindernisse
- Baseline für Vergleiche

### 2. Zufälliger Start (`random_start`)
- Start-Position wird zufällig gewählt
- Ziel und Hindernisse bleiben fest

### 3. Zufälliges Ziel (`random_goal`)
- Ziel-Position wird zufällig gewählt
- Start und Hindernisse bleiben fest

### 4. Zufällige Hindernisse (`random_obstacles`)
- Hindernis-Positionen werden zufällig gewählt
- Start und Ziel bleiben fest

### 5. Container-Umgebung (`container`)
- Pickup/Dropoff-Aufgaben
- Komplexere Reward-Struktur

## 📈 Ergebnisse

Nach dem Training werden folgende Dateien erstellt:

**Q-Learning:**
- **Q-Tabellen**: `q_table_{scenario}.npy`
- **Lernkurven**: `exports/learning_curve_{scenario}.png`
- **Erfolgsraten**: `exports/success_curve_{scenario}.png`

**Deep Q-Learning:**
- **Modelle**: `dqn_model_{scenario}.pth`
- **Trainingsverlauf**: `exports/dqn_training_{scenario}.pdf`
- **Verlustkurven**: Integriert in Trainingsplots

**Vergleiche:**
- **Algorithmus-Vergleich**: `exports/algorithm_comparison.pdf`
- **Detaillierte CSV**: `exports/algorithm_comparison.csv`
- **Heatmaps**: `exports/algorithm_heatmap_comparison.pdf`

## 🔧 Troubleshooting

### Häufige Probleme

**ModuleNotFoundError:**
```bash
pip install -r requirements.txt
```

**CUDA/GPU Probleme (DQN):**
- DQN erkennt automatisch verfügbare Hardware
- Bei Problemen: CPU-Modus in `config.py` forcieren

**Memory Errors (DQN):**
- Reduziere `DQN_BATCH_SIZE` oder `DQN_BUFFER_SIZE` in `config.py`

**Import Errors:**
- Führe Befehle aus dem Projekt-Root aus: `python -m src.dqn.train`