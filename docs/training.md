# Training und Konfiguration

## Trainingssteuerung

Das Training erfolgt über skriptgesteuerte Abläufe in den jeweiligen Unterordnern `q_learning/` und `dqn/`. Für jedes Verfahren existiert ein zentrales Trainingsskript (`train_all_scenarios.py`), das mehrere Umgebungsmodi sequentiell durchläuft. Dabei wird jeder Durchlauf mehrfach wiederholt.

Das DQN-Training erfolgt objektorientiert über die Klasse `DQNTrainer`, während Q-Learning über Subprozess-Aufrufe gesteuert wird.

## Konfigurationsprinzip

Die wichtigsten Hyperparameter sind zentral in `config.py` hinterlegt. Zugriff und Validierung erfolgen über `config_utils.py`. Parameter wie Lernrate, Epsilon-Strategie, Episodenanzahl oder Seed können über diese Dateien angepasst werden. 

Zur flexiblen Ausführung lassen sich bestimmte Parameter zusätzlich über Kommandozeilenargumente oder Umgebungsvariablen übergeben.

## Trainingsmetriken

Während und nach dem Training werden verschiedene Kennzahlen erfasst, unter anderem:

- Erfolgsrate über alle Episoden
- Durchschnittliche Schrittanzahl bis zum Ziel
- Durchschnittlicher Reward pro Episode
- Explorationseinstellungen und Replay-Speichergröße (bei DQN)

Die Trainingsverläufe werden pro Szenario als PDF-Dateien gespeichert und zusätzlich tabellarisch dokumentiert.

## Ablaufbeispiel

```bash
cd src/q_learning
python train_all_scenarios.py
```

```bash
cd src/dqn
python train_all_scenarios.py --episodes 500 --runs 3
```
