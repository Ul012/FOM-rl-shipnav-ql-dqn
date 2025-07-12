# Funktionsweise der Algorithmen

## Q-Learning

- Verwendung einer Q-Tabelle zur Entscheidung
- Für jeden Zustand-Aktions-Paar wird ein Q-Wert gespeichert, der iterativ aktualisiert wird.
- Aktualisierung mittels Bellman-Gleichung
- Exploration über feste oder decaying ε-greedy-Strategie
- Ziel ist die Annäherung an eine optimale Policy, die in jeder Situation die langfristig beste Entscheidung ermöglicht.

Dabei stehen:
- `α` für die Lernrate
- `γ` für den Diskontfaktor
- `r` für die beobachtete Belohnung
- `s'` für den Folgezustand

### Entscheidungsstrategie

Zur Auswahl der nächsten Aktion wird typischerweise eine Epsilon-Greedy-Strategie verwendet. Diese kombiniert zufällige Exploration mit der Ausnutzung des aktuellen Wissens.

### Zustandsrepräsentation

Die Zustände werden diskret codiert. Je nach Umgebung bestehen sie aus Positionsinformationen (z. B. Zeile, Spalte) sowie bei erweiterten Umgebungen zusätzliche Merkmale wie den Transportstatus eines Objekts.

## Deep Q-Learning (DQN)

- Verwendung eines neuronalen Netzwerks zur Approximation der Q-Funktion
- Techniken: Experience Replay, Target Network
- Training mittels Mini-Batches und MSE-Loss
- Unterstützt kontinuierlichere Zustandsräume als tabellarisches Q-Learning

### Hauptkomponenten

1. **Q-Network**: Vorwärtspass zur Schätzung der Q-Werte
2. **Target Network**: Stabilisierte Zielwerte durch verzögertes Update
3. **Experience Replay**: Training auf zufälligen Mini-Batches aus einer Replay-Memory

### Lernprozess

Der Trainingsprozess basiert auf der Minimierung des Fehlers zwischen dem geschätzten Q-Wert und einem Zielwert:

Zur Trainingszeit wird ein Zielwert berechnet, auf den sich das Q-Netzwerk iterativ anpasst. Die Netzwerkarchitektur kann flexibel angepasst werden und besteht typischerweise aus mehreren vollständig verbundenen Schichten mit ReLU-Aktivierung.

## Vergleich der Ansätze

| Eigenschaft           | Q-Learning              | Deep Q-Learning         |
|-----------------------|-------------------------|--------------------------|
| Q-Funktion            | Tabelle                 | Neuronales Netzwerk      |
| Speicherbedarf        | Zustands-Aktions-Tabelle | Modellparameter          |
| Aktualisierung        | Einzelne Updates        | Batch-Training mit Replay |
| Zustandsraumgröße     | Klein, diskret          | Groß, kontinuierlich     |
| Konvergenzverhalten   | Theoretisch garantiert  | Approximativ, stabilisiert durch Target-Netzwerke |

## Gemeinsame Systemkomponenten

### Umgebungen

Beide Algorithmen arbeiten mit denselben simulierten Umgebungen:
- Einfache Gitter-Navigation
- Erweiterte Aufgaben mit Container-Objekt

### Belohnungsstruktur

Belohnungen werden in Abhängigkeit der Agenteninteraktion mit der Umgebung vergeben. Belohnungsarten sind unter anderem:
- Bewegungskosten
- Zielerreichung
- Hindernis-Kollision
- Abbruch bei Wiederholungen
- Aufnehmen und Abliefern von Objekten

Die genauen Werte sind konfigurierbar und können über die zentrale Konfigurationsdatei angepasst werden.

### Terminierungskriterien

Eine Episode endet unter folgenden Bedingungen:
- Ziel wurde erreicht
- Ein Abbruchkriterium wurde erfüllt (z. B. zu viele Schleifen)
- Ein Hindernis wurde getroffen
- Die maximale Schrittanzahl wurde überschritten

## Szenarien

Zur Evaluation werden verschiedene Umgebungsmodi verwendet. Diese variieren beispielsweise durch:
- Fixe oder zufällige Start- und Zielpositionen
- Platzierung von Hindernissen
- Erweiterte Aufgaben wie Objekthandling

Die Szenarien lassen sich modular anpassen und sind zentral konfigurierbar.
