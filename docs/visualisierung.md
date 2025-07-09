# Visualisierung und Analyse

## Q-Learning

### Policy-Darstellung

Für Q-Learning kann die erlernte Policy grafisch visualisiert werden. Dies erfolgt durch eine animierte Darstellung der Agentenentscheidungen innerhalb der Gitterumgebung. Zusätzlich stehen Pfeildiagramme und Zustandsübersichten zur Verfügung.

Pygame-Animation mit Emojis:
- 🚢 Agent/Schiff
- 🧭 Start, 🏁 Ziel, 🪨 Hindernis
- 📦 Pickup (Container-Szenario)
- ↑→↓← Policy-Pfeile

### Evaluation

Zur Auswertung des Trainings können verschiedene Statistiken erzeugt werden:
- Erfolgsraten über alle Episoden
- Belohnungsverteilung
- Gründe für das Beenden von Episoden (z. B. Ziel erreicht, Timeout, Kollision)

### Szenarienvergleich

Es ist möglich, die Ergebnisse aus mehreren Szenarien gegenüberzustellen. Dabei werden Erfolgskennzahlen und Abbrucharten analysiert und visualisiert.

### Q-Tabellen-Inspektion

Für weiterführende Analysen kann die trainierte Q-Tabelle interaktiv untersucht werden. Die Darstellung kann je nach Anwendungsfall angepasst werden, etwa als Matrixansicht oder als komprimierte Übersicht.

## Deep Q-Learning (DQN)

### Trainingsvisualisierung

Beim DQN-Training werden automatisch verschiedene Plots erzeugt, etwa:
- Entwicklung des Trainingsverlusts (Loss)
- Verlauf der durchschnittlichen Belohnung
- Erfolgsrate im Zeitverlauf
- Zusammenfassende Mehrfachdarstellungen

### Evaluation

Analog zum Q-Learning kann auch die DQN-Policy visualisiert und analysiert werden. Die Darstellung erfolgt konsistent mit der Q-Learning-Ansicht zur Vergleichbarkeit.

### Szenarienvergleich

Ergebnisse aus mehreren Szenarien lassen sich aggregieren und vergleichen. Neben Erfolgskennzahlen können auch statistische Metriken und Hardware-Auslastungen dokumentiert werden.

## Algorithmusvergleich

Zur Gegenüberstellung von Q-Learning und DQN werden standardisierte Visualisierungen erstellt. Dazu zählen:

- Durchschnittliche Erfolgsraten pro Szenario
- Durchschnittliche Episodenschritte
- Verteilungen der erzielten Belohnungen
- Differenzanalysen zwischen beiden Methoden

Die Ergebnisse werden tabellarisch und grafisch zusammengeführt und ermöglichen eine differenzierte Bewertung der Algorithmen.

## Analyseexporte

Während Training und Evaluation entstehen automatisch exportierte Ausgabedateien wie:
- PDF-Diagramme
- CSV-Tabellen
- Screenshots von Agentenverläufen

Diese dienen der weiteren Analyse, der Dokumentation und dem Vergleich von Trainingsverläufen.

## Konfiguration

Das Visualisierungsverhalten kann über Parameter gesteuert werden:
- Aktivierung oder Deaktivierung interaktiver Fenster
- Animationsgeschwindigkeit
- Exportformat (z. B. nur PDF statt Live-Plot)

Diese Einstellungen werden zentral in der Konfigurationsdatei verwaltet und gelten für beide Algorithmen.

## Systemarchitektur

Visualisierungen greifen auf gemeinsame Module zurück, etwa zur Anzeige, zur Formatierung und zur Verwaltung von Ausgabepfaden. Unterschiede zwischen Q-Learning und DQN werden intern behandelt, ohne dass sich dies auf die Bedienung auswirkt.
