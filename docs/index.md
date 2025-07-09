# 🚢 Q-Learning vs Deep Q-Learning für Schiffsnavigation

Dieses Projekt vergleicht zwei verstärkendes Lernen (Reinforcement Learning) basierte Ansätze zur autonomen Navigation in einer simulierten Schiffsumgebung: das klassische tabellenbasierte Q-Learning und Deep Q-Learning (DQN) mit neuronalen Netzwerken.

## 🎯 Projektziele

- Implementierung beider Algorithmen zur Navigation in diskreten Gitterumgebungen
- Vergleich des Lernverhaltens unter einheitlichen Bedingungen
- Evaluation in verschiedenen Szenarien mit zunehmender Komplexität
- Bereitstellung einer modularen, reproduzierbaren Projektstruktur

## 🧠 Algorithmenüberblick

### Q-Learning
- Verwendet eine Q-Tabelle zur Speicherung von Zustands-Aktions-Werten
- Entscheidung basierend auf einer Epsilon-Greedy-Strategie
- Eignet sich für überschaubare, vollständig diskrete Zustandsräume

### Deep Q-Learning (DQN)
- Approximation der Q-Funktion durch ein neuronales Netzwerk
- Nutzung von Techniken wie Experience Replay und Target Networks
- Ermöglicht Anwendung auf größere oder kontinuierliche Zustandsräume

## 🗺️ Szenarien

Die Algorithmen werden in mehreren vorgegebenen Umgebungsvarianten getestet. Diese unterscheiden sich durch Start- und Zielbedingungen sowie durch zusätzliche Aufgaben wie das Aufnehmen und Abliefern eines Containers:

- **Statisches Szenario**: Feste Start- und Zielposition
- **Zufälliger Start**: Startposition wird zufällig gewählt
- **Zufälliges Ziel**: Zielposition variiert
- **Zufällige Hindernisse**: Platzierung von Hindernissen ändert sich
- **Container-Szenario**: Der Agent muss zusätzlich einen Container transportieren

## 🧱 Projektstruktur

\`\`\`plaintext
ship-navigation-ql-dqn/
├── src/
│   ├── q_learning/          # Q-Learning-Implementierung
│   ├── dqn/                 # DQN-Implementierung
│   ├── comparison/          # Vergleich beider Algorithmen
│   └── shared/              # Gemeinsame Konfiguration und Umgebungen
├── docs/                    # Technische und inhaltliche Dokumentation
└── exports/                 # Automatisch erzeugte Modelle und Statistiken
\`\`\`

## 🔍 Weiterführende Inhalte

- [⚙️ Setup](setup.md) – Installationsanleitung und Projektstart
- [🧠 Funktionsweise](funktionsweise.md) – Technischer Überblick über die Algorithmen
- [🎯 Training](training.md) – Ausführung, Konfiguration und Szenariensteuerung
- [📊 Visualisierung](visualisierung.md) – Darstellung von Policies und Evaluationen
- [📚 Entwicklung](dokumentation.md) – Informationen zur Projektstruktur und Pflege
