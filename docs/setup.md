# Installation und Setup

## Systemanforderungen

- Python ab Version 3.8
- Mindestens 4 GB Arbeitsspeicher
- Optional: GPU-Unterstützung (für Deep Q-Learning)

## Installation

### Repository klonen und vorbereiten

```bash
git clone https://github.com/Ul012/FOM-rl-shipnav-ql-dql.git
cd FOM-rl-shipnav-ql-dql
```

### Virtuelle Umgebung erstellen

```bash
python -m venv ql-dqn-venv
# Aktivierung (je nach Betriebssystem)
ql-dqn-venv\Scripts\activate       # Windows
source ql-dqn-venv/bin/activate      # Linux/macOS
```

### Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### Funktionsprüfung

```bash
cd src
python -c "from shared.envs import GridEnvironment; print('Setup OK')"
```

## Projektstruktur

```plaintext
ship-navigation-ql-dqn/
├── src/                   # Implementierung von Q-Learning und DQN
│   ├── q_learning/        # Q-Learning
│   ├── dqn/               # Deep Q-Learning
│   ├── comparison/        # Vergleich beider Ansätze
│   └── shared/            # Gemeinsame Umgebungen und Konfiguration
├── docs/                  # Dokumentation
└── exports/               # Trainings- und Evaluationsausgaben
```

## Verwendung

### Q-Learning

- Training einzelner oder aller Szenarien
- Evaluation und Visualisierung über eigene Skripte
- Export der Ergebnisse erfolgt automatisch

### Deep Q-Learning

- Steuerung über Parameter (z. B. Modus, Episodenzahl)
- Modelle können trainiert oder zur reinen Evaluation geladen werden
- Visualisierung und Analyse wie bei Q-Learning

### Vergleich

Ein zentrales Skript ermöglicht den Vergleich beider Algorithmen unter einheitlichen Bedingungen.

## Fehlerbehandlung

Typische Hinweise zur Problembehebung:
- Sicherstellen, dass Skripte im `src`-Verzeichnis ausgeführt werden
- Abhängigkeiten prüfen, insbesondere PyTorch und pygame
- Exportverzeichnisse bei Bedarf manuell anlegen
