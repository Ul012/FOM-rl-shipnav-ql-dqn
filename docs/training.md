# Training und Konfiguration

## Q-Learning

### Trainingsablauf

Das Q-Learning-Training erfolgt durch iteratives Ausführen von Episoden in einer simulierten Umgebung. Dabei werden die Q-Werte nach jedem Schritt entsprechend der Lernregel angepasst.

Typischer Ablauf:
- Start der Umgebung im gewünschten Szenario
- Schrittweises Lernen durch Interaktion mit der Umgebung
- Speicherung der Q-Tabelle und optionaler Auswertungen

Es können einzelne oder mehrere vordefinierte Szenarien nacheinander trainiert werden. Die Trainingsparameter werden zentral über eine Konfigurationsdatei gesteuert.

## Deep Q-Learning (DQN)

### Trainingsablauf

Auch das DQN-Training erfolgt episodisch. Die Aktualisierung der Netzwerkgewichte basiert auf gesampelten Zustandsübergängen aus einem Replay-Puffer. Zusätzlich werden ein Zielnetzwerk und batchweises Lernen verwendet.

Wichtige Merkmale:
- Netzwerkstruktur und Lernparameter sind konfigurierbar
- Training kann mit oder ohne Visualisierung durchgeführt werden
- Es besteht die Möglichkeit, vortrainierte Modelle zu laden und nur zu evaluieren

## Algorithmusvergleich

Ein separater Vergleichsmodus ermöglicht die parallele oder sequenzielle Auswertung beider Algorithmen. Dabei werden gemeinsame Metriken erfasst, etwa Erfolgsraten, durchschnittliche Schritte oder kumulierte Belohnungen. Die Ergebnisse werden tabellarisch und grafisch zusammengeführt.

## Konfigurierbare Parameter

Die wichtigsten Einstellungen für Training und Evaluation werden in einer zentralen Konfigurationsdatei verwaltet. Diese umfasst unter anderem:

- Wahl des Szenarios
- Anzahl der Trainings-Episoden
- Maximale Schrittanzahl pro Episode
- Parameter für Exploration und Lernrate
- Netzwerkspezifische Hyperparameter (DQN)

Diese Werte sind jederzeit anpassbar und ermöglichen eine flexible Steuerung des Experiments.

## Szenarien

Das System unterstützt mehrere Umgebungsvarianten, die sich in Start-/Zielbedingungen und Aufgabenstellung unterscheiden. Typische Szenarien umfassen:

- **Statisches Layout**: Feste Start- und Zielpositionen
- **Zufälliger Start/Ziel**: Dynamische Start- oder Zielsetzung
- **Variable Hindernisse**: Zufällige Platzierung von Blockaden
- **Erweiterte Aufgaben**: Transport eines Objekts zum Zielort

Jeder dieser Modi kann einzeln oder im Batch trainiert und evaluiert werden.

## Export und Evaluation

Während des Trainings und der anschließenden Auswertung werden automatisch verschiedene Ergebnisse gespeichert:

- Lernkurven (z. B. Erfolgsrate über Episoden)
- Ausgewertete Modelle oder Q-Tabellen
- Vergleichsdaten (z. B. CSVs oder Diagramme)

Diese Dateien dienen der Analyse, dem Vergleich und der Reproduzierbarkeit der Ergebnisse.

## Reproduzierbarkeit

Für konsistente Ergebnisse werden alle relevanten Zufallsquellen kontrolliert. Ein globaler Seed stellt sicher, dass identische Konfigurationen wiederholbare Resultate liefern – sowohl für Q-Learning als auch für DQN. Unterschiede in der verwendeten Hardware (CPU/GPU) werden berücksichtigt.
