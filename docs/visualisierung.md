# Visualisierung und Analyse

## Q-Learning Visualisierung

### Policy-Darstellung
```bash
cd src/q_learning
python visualize_policy.py
```

Pygame-Animation mit Emojis:
- 🚢 Agent/Schiff
- 🧭 Start, 🏁 Ziel, 🪨 Hindernis
- 📦 Pickup (Container-Szenario)
- ↑→↓← Policy-Pfeile

### Evaluation
```bash
python evaluate_policy.py
```

Erstellt:
- Erfolgsraten-Balkendiagramm
- Reward-Histogramm
- Terminierungsarten-Verteilung

### Szenarien-Vergleich
```bash
python compare_scenarios.py
```

Erstellt:
- Erfolgsraten aller Q-Learning Szenarien
- Terminierungsarten-Analyse
- Statistische Vergleichstabelle

### Q-Tabellen-Inspektion
```bash
python inspect_q_tables.py
```

Interaktive Optionen:
1. Aktuelles Szenario
2. Spezifisches Szenario
3. Alle Q-Tabellen
4. Formen-Vergleich
5. Matrix-Darstellung

## DQN Visualisierung

### Training-Plots
DQN Training erstellt automatisch:
- Lernkurven mit Moving Average
- Erfolgskurven mit Exploration Decay
- Loss-Entwicklung
- 4-Panel Statistik-Übersicht

### Evaluation
```bash
cd src/dqn
python train.py --mode static --eval-only
```

Zeigt gelernte DQN Policy mit denselben Emojis.

### Multi-Szenario
```bash
python train_all_scenarios.py
```

Erstellt:
- Performance-Vergleiche aller DQN Szenarien
- Hardware-Metriken (GPU/CPU)
- Batch-Statistiken

## Algorithmus-Vergleich

```bash
cd src/comparison
python compare_algorithms.py
```

### Vergleichs-Visualisierungen

1. **Performance-Vergleich**
   - Erfolgsraten: Q-Learning vs DQN
   - Durchschnittliche Schritte
   - Durchschnittliche Belohnungen
   - Error Bars über mehrere Runs

2. **Heatmaps**
   - Q-Learning Performance pro Szenario
   - DQN Performance pro Szenario
   - Direkte Unterschiede

3. **Box Plots**
   - Erfolgsrate-Verteilungen
   - Schritte-Verteilungen
   - Reward-Verteilungen

## Export-Dateien

### Q-Learning (`src/q_learning/exports/`)
```
train_learning_curve.pdf
train_success_curve.pdf
train_statistics.pdf
evaluate_policy_success_rate.pdf
evaluate_policy_reward_histogram.pdf
success_rates.pdf
failure_modes.pdf
```

### DQN (`src/dqn/exports/`)
```
dqn_training_{scenario}.pdf
dqn_all_scenarios_summary.csv
dqn_all_scenarios_comparison.pdf
```

### Comparison (`src/comparison/exports/`)
```
algorithm_comparison.pdf
algorithm_comparison.csv
algorithm_comparison_stats.csv
algorithm_heatmap_comparison.pdf
```

### Screenshots
```
exports/agent_final_position.png
```

## Interaktive Features

### Q-Learning
- Echtzeit Q-Tabelle Animation
- Schritt-für-Schritt Logging (Position, Q-Werte, Reward)
- Automatische Screenshots

### DQN
- Neural Network Entscheidungs-Animation
- Experience Replay Status
- Network-Output Logging

### Vergleich
- Algorithm Toggle zwischen Policies
- Side-by-Side Visualisierung
- Performance Overlay

## Konfiguration

```python
# In shared/config.py
EXPORT_PDF = True                    # PDF-Export aktivieren
CELL_SIZE = 80                       # Pygame-Zellgröße
FRAME_DELAY = 0.4                    # Animationsgeschwindigkeit
SHOW_VISUALIZATIONS = True           # Interaktive Plots

# Algorithmus-spezifische Export-Pfade
Q_LEARNING_EXPORT_PATH = "exports/"  # → src/q_learning/exports/
DQN_EXPORT_PATH = "exports/"         # → src/dqn/exports/
COMPARISON_EXPORT_PATH = "exports/"  # → src/comparison/exports/
```

## Performance-Optimierung

### Für schnelle Verarbeitung
```python
SHOW_VISUALIZATIONS = False          # Nur PDF-Export
EXPORT_PDF = True                    # Für spätere Analyse
```

### Für interaktive Analyse
```python
SHOW_VISUALIZATIONS = True           # Matplotlib-Fenster
FRAME_DELAY = 0.2                    # Schnellere Animation
```

## Technische Details

### Gemeinsame Komponenten
- Pygame Rendering (identisch für beide Algorithmen)
- PDF Export-Management
- Matplotlib Styling

### Algorithmus-spezifisch
- **Q-Learning**: `q_learning/utils/visualization.py`
- **DQN**: PyTorch-kompatible Plots
- **Shared**: Einheitliche Formatierung