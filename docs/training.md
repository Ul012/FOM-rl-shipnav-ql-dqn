# Training und Konfiguration

## Q-Learning Training

### Einzelszenario
```bash
cd src/q_learning
python train.py
```

Erstellt:
- Q-Tabelle: `exports/q_table_{szenario}.npy`
- Lernkurven und Erfolgsstatistiken als PDF

### Alle Szenarien
```bash
python train_all_scenarios.py
```

Optionen:
- Szenario-Auswahl (einzeln/alle)
- Visualisierungsmodus (interaktiv/automatisiert)
- Training-Modus (sequenziell/parallel)

## Deep Q-Learning Training

### Einzelszenario
```bash
cd src/dqn

# Training
python train.py --mode static --episodes 500
python train.py --mode container --episodes 300

# Nur Evaluation (lädt Modell)
python train.py --mode static --eval-only

# Ohne Plots
python train.py --mode static --episodes 200 --no-plot
```

Erstellt:
- Modell: `exports/dqn_model_{szenario}.pth`
- Training-Plots mit Loss-Kurven

### Alle Szenarien
```bash
# Standard
python train_all_scenarios.py

# Angepasst
python train_all_scenarios.py --episodes 300 --runs 2
python train_all_scenarios.py --scenarios static container
```

## Algorithmus-Vergleich

```bash
cd src/comparison

# Vollständig
python compare_algorithms.py --runs 5

# Test
python compare_algorithms.py --ql-episodes 100 --dqn-episodes 100 --runs 2

# Spezifisch
python compare_algorithms.py --scenarios static random_start
```

Erstellt:
- Vergleichsplots und Heatmaps
- CSV-Dateien mit Statistiken
- Performance-Analysen

## Parameter

### Zentrale Konfiguration (`shared/config.py`)

```python
# Umgebung
ENV_MODE        # Szenario-Auswahl
EPISODES        # Q-Learning Episoden
DQN_EPISODES    # DQN Episoden
MAX_STEPS       # Schritte pro Episode
SEED            # Reproduzierbarkeit

# Q-Learning
ALPHA           # Lernrate
GAMMA           # Diskontfaktor
EPSILON         # Explorationsrate

# DQN
DQN_LEARNING_RATE       # Neural Network Lernrate
DQN_BATCH_SIZE          # Batch-Größe
DQN_BUFFER_SIZE         # Experience Replay Puffer
DQN_HIDDEN_SIZE         # Network-Architektur
DQN_TARGET_UPDATE_FREQ  # Target Network Updates

```

### Export-Pfade
```python
# Algorithmus-spezifische Exports
Q_LEARNING_EXPORT_PATH = "exports/"      # → src/q_learning/exports/
DQN_EXPORT_PATH = "exports/"             # → src/dqn/exports/
COMPARISON_EXPORT_PATH = "exports/"      # → src/comparison/exports/
```

## Szenarien

| Szenario | ENV_MODE | Q-Learning | DQN |
|----------|----------|------------|-----|
| Statisch | `"static"` | ✅ | ✅ |
| Zufälliger Start | `"random_start"` | ✅ | ✅ |
| Zufälliges Ziel | `"random_goal"` | ✅ | ✅ |
| Zufällige Hindernisse | `"random_obstacles"` | ✅ | ✅ |
| Container | `"container"` | ✅ | ✅ |

## Generierte Dateien

### Q-Learning (`src/q_learning/exports/`)
```
q_table_{scenario}.npy               # Trainierte Q-Tabellen
train_learning_curve.pdf             # Lernkurven
train_success_curve.pdf              # Erfolgsraten
train_statistics.pdf                 # Statistiken
evaluate_policy_*.pdf                # Evaluation
```

### DQN (`src/dqn/exports/`)
```
dqn_model_{scenario}.pth             # Trainierte Modelle
dqn_training_{scenario}.pdf          # Training-Plots
dqn_all_scenarios_summary.csv        # Zusammenfassung
dqn_all_scenarios_comparison.pdf     # Vergleiche
```

### Comparison (`src/comparison/exports/`)
```
algorithm_comparison.pdf             # Hauptvergleich
algorithm_comparison.csv             # Daten
algorithm_heatmap_comparison.pdf     # Heatmaps
```

## Reproduzierbarkeit

### Seed-Kontrolle
- Zentrale Seed-Kontrolle über `SEED` in `shared/config.py`
- Deterministisches Verhalten bei gleichem Seed
- Reproduzierbare Ergebnisse für beide Algorithmen

### Hardware
- **Q-Learning**: CPU-optimiert
- **DQN**: GPU-Support (automatisch erkannt)

## Troubleshooting

### Q-Learning
- Niedrige Erfolgsrate → ALPHA, EPSILON erhöhen
- Langsame Konvergenz → Mehr EPISODES

### DQN
- Nicht konvergierend → Lernrate reduzieren
- Memory Errors → Batch/Buffer Size reduzieren
- Langsam → GPU aktivieren

### Allgemein
- Import Errors → Aus `src/` ausführen
- Fehlende Exports → Export-Ordner erstellen