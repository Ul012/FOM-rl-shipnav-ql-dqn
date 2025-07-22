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
### Hinweis zur Ausführung

Die Trainings- und Vergleichsskripte sind auf eine Ausführung direkt aus der Entwicklungsumgebung (z. B. über den „Run“-Button in PyCharm) optimiert. Dabei wird die Projektstruktur korrekt erkannt und alle Importe funktionieren ohne weitere Anpassungen.

Für die Ausführung über das Terminal sind ggf. zusätzliche Konfigurationsschritte erforderlich (z. B. `PYTHONPATH` oder Modulaufruf mit `-m`).

Empfohlen wird daher die Nutzung der bereitgestellten Run-Konfigurationen in PyCharm.


## 📊 Vergleichsvisualisierung

Es stehen drei Varianten für den visuellen Vergleich der Algorithmen zur Verfügung:

1. **Overview-Visualisierung** (`1_compare_algorithms_overview.py`)  
   → Führt die Evaluation durch und speichert die CSV-Datei (`algorithm_comparison_overview.csv`)

2. **2x2 V1** (`compare_algorithms_scientific.py`)  
   → Wissenschaftliches Grid-Layout mit Erfolgsraten, Belohnung, Schritten und Scatterplot  
   **Nutzt die CSV aus Overview-Variante als Grundlage.**

3. **2x2 V2** (`compare_algorithms_heatmap.py`)  
   → Fokus auf Darstellung mit Heatmap  
   **Nutzt ebenfalls die CSV aus Overview-Variante als Grundlage.**

➡️ **Wichtig:** Die Overview-Variante muss vor den anderen beiden ausgeführt werden.



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

