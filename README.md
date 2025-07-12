# 🚢 Reinforcement Learning: Q-Learning vs Deep Q-Learning in Grid Navigation

Dieses Projekt untersucht und vergleicht tabellenbasiertes Q-Learning mit Deep Q-Learning (DQN) zur Navigation autonomer Agenten in simulierten Gitterumgebungen. Ziel ist eine reproduzierbare Evaluation beider Verfahren unter einheitlichen Rahmenbedingungen.

## 📁 Projektstruktur

```
ship-navigation-ql-dqn/
├── src/
│   ├── q_learning/          # Q-Learning-Training, Evaluation, Visualisierung
│   ├── dqn/                 # DQN-Training, Evaluation, Visualisierung
│   ├── comparison/          # Algorithmusvergleich & Visualisierungen
│   └── shared/              # Konfigurationen, Umgebungen, utils
├── docs/                    # Dokumentation (MkDocs)
├── exports/                 # Ausgabedateien (PDFs, CSVs, Plots)
└── README.md                # Projektübersicht
```

## ⚙️ Setup

1. Repository klonen  
2. Virtuelle Umgebung erstellen und aktivieren  
3. Abhängigkeiten installieren

```bash
git clone https://github.com/DeinUser/ship-navigation-ql-dqn.git
cd ship-navigation-ql-dqn
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

## 🧠 Trainingssteuerung

### Q-Learning

```bash
cd src/q_learning
python train_all_scenarios.py
```

### Deep Q-Learning

```bash
cd src/dqn
python train_all_scenarios.py --episodes 500 --runs 3
```

### Vergleichsvisualisierung
## 📊 Vergleichsvisualisierung

Es stehen drei Varianten für den visuellen Vergleich der Algorithmen zur Verfügung:

1. **2x3-Visualisierung** (`compare_algorithms_2x3.py`)  
   → Führt die Evaluation durch und speichert die CSV-Datei (`algorithm_comparison_2x3.csv`)

2. **2x2 V1** (`compare_algorithms_2x2_v1.py`)  
   → Wissenschaftliches Grid-Layout mit Erfolgsraten, Belohnung, Schritten und Scatterplot  
   **Nutzt die CSV aus 2x3 als Grundlage.**

3. **2x2 V2** (`compare_algorithms_2x2_v2.py`)  
   → Fokus auf Single-Agent-Darstellung mit Heatmap  
   **Nutzt ebenfalls die CSV aus 2x3 als Grundlage.**

➡️ **Wichtig:** Die 2x3-Variante muss vor den anderen beiden ausgeführt werden.


```bash
cd src/comparison
python compare_algorithms_2x2_v1.py
```

## 🌍 Szenarien

Es werden fünf Varianten unterschieden:
- `static` – feste Start-/Ziel-/Hindernispositionen
- `random_start` – zufälliger Start
- `random_goal` – zufälliges Ziel
- `random_obstacles` – zufällige Hindernisse
- `container` – Aufgaben mit Pickup & Dropoff

## 📊 Ergebnisse

Ergebnisse und Visualisierungen (Lernkurven, Erfolgsraten, Vergleichsplots) werden automatisch im jeweiligen `exports/`-Verzeichnis gespeichert. Q-Tabellen und Modellgewichte werden szenariobezogen abgelegt.

## 📚 Dokumentation

Die technische und inhaltliche Dokumentation ist mit MkDocs aufbereitet:

```bash
mkdocs serve
```

→ erreichbar unter [http://localhost:8000](http://localhost:8000)

