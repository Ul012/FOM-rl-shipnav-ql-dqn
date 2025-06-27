# RL Ship Navigation: Q-Learning vs Deep Q-Learning

Ein Vergleich zwischen Q-Learning und Deep Q-Learning Algorithmen für autonome Schiffsnavigation in Grid-Umgebungen.

## 📋 Projektübersicht

Dieses Projekt vergleicht die Leistung von klassischem Q-Learning und Deep Q-Learning (DQN) in verschiedenen Navigationsszenarien:

- **Statische Umgebung**: Feste Start-, Ziel- und Hindernis-Positionen
- **Dynamische Szenarien**: Zufällige Start-, Ziel- oder Hindernis-Positionen  
- **Container-Umgebung**: Pickup/Dropoff-Aufgaben

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
│   ├── dqn/             # Deep Q-Learning Implementation (in Entwicklung)
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

### Konfiguration anpassen

Editiere `src/shared/config.py` für:
- Hyperparameter (Lernrate, Epsilon, Gamma)
- Environment-Settings (Grid-Größe, Rewards)
- Training-Parameter (Episoden, Max-Steps)

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

- **Q-Tabellen**: `q_table_{scenario}.npy`
- **Lernkurven**: `exports/learning_curve_{scenario}.png`
- **Erfolgsraten**: `exports/success_curve_{scenario}.png`
- **Vergleichsreports**: `exports/comparison_report.pdf`
- 
## 🧪 Experimente

### Aktuell verfügbar:
- ✅ Q-Learning für alle Szenarien
- ✅ Hyperparameter-Tuning
- ✅ Visualisierung und Evaluation

### In Entwicklung:
- 🚧 Deep Q-Learning (DQN) Implementation
- 🚧 Direkter Algorithmus-Vergleich
- 🚧 Performance-Benchmarks
- 🚧 Mkdocs zur Dokumentation
