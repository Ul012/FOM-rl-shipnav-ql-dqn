# Installation und Setup

## Systemvoraussetzungen

- Python 3.8+
- 4 GB RAM
- Optional: CUDA-GPU für DQN

## Installation

```bash
# 1. Repository klonen
git clone https://github.com/Ul012/FOM-rl-shipnav-ql-dql.git
cd FOM-rl-shipnav-ql-dql

# 2. Virtual Environment
python -m venv ql-dqn-venv
ql-dqn-venv\Scripts\activate  # Windows
source ql-dqn-venv/bin/activate  # Linux/Mac

# 3. Dependencies
pip install -r requirements.txt

# 4. Verifikation
cd src
python -c "from shared.envs import GridEnvironment; print('Setup OK')"
```

## Projektstruktur

```
ship-navigation-ql-dqn/
├── src/
│   ├── q_learning/          # Q-Learning Scripts + exports/
│   ├── dqn/                 # DQN Scripts + exports/
│   ├── comparison/          # Vergleichs-Scripts + exports/
│   └── shared/              # Gemeinsame Konfiguration + Environments
├── docs/                    # Diese Dokumentation
└── requirements.txt
```

## Verwendung

### Q-Learning

```bash
cd src/q_learning

# Einzelszenario
python train.py

# Alle Szenarien
python train_all_scenarios.py

# Evaluation
python evaluate_policy.py

# Visualisierung
python visualize_policy.py
```

### Deep Q-Learning

```bash
cd src/dqn

# Training
python train.py --mode static --episodes 500
python train.py --mode container --episodes 300

# Nur Evaluation
python train.py --mode static --eval-only

# Alle Szenarien
python train_all_scenarios.py --episodes 300 --runs 2
```

### Algorithmus-Vergleich

```bash
cd src/comparison

# Vollständiger Vergleich
python compare_algorithms.py --runs 3

# Schneller Test
python compare_algorithms.py --ql-episodes 100 --dqn-episodes 100 --runs 1
```

## Troubleshooting

### Häufige Probleme

**ModuleNotFoundError:**
```bash
cd src  # Immer aus src/ ausführen
python -c "from shared.config import EPISODES; print('OK')"
```

**PyTorch (DQN) Probleme:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Bei GPU-Problemen: CPU-Modus in config.py forcieren
```

**Memory-Probleme (DQN):**
```python
# In shared/config.py reduzieren:
DQN_BATCH_SIZE = 16
DQN_BUFFER_SIZE = 2000
```

**Pygame nicht sichtbar:**
```bash
pip install --upgrade pygame
```

### Export-Verzeichnisse

Falls Ordner fehlen:
```bash
mkdir -p src/q_learning/exports
mkdir -p src/dqn/exports
mkdir -p src/comparison/exports
```