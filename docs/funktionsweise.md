# Funktionsweise der Algorithmen

## Q-Learning

Q-Learning ist ein tabellenbasierter Ansatz zur Lösung von Markov-Entscheidungsprozessen. Für jeden Zustand-Aktions-Paar wird ein Q-Wert gespeichert, der iterativ aktualisiert wird. Ziel ist die Annäherung an eine optimale Policy, die in jeder Situation die langfristig beste Entscheidung ermöglicht.

### Lernformel

```python
Q(s,a) ← Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
```

Dabei stehen:
- `α` für die Lernrate
- `γ` für den Diskontfaktor
- `r` für die beobachtete Belohnung
- `s'` für den Folgezustand

### Entscheidungsstrategie

Zur Auswahl der nächsten Aktion wird typischerweise eine Epsilon-Greedy-Strategie verwendet. Diese kombiniert zufällige Exploration mit der Ausnutzung des aktuellen Wissens.

### Zustandsrepräsentation

Die Zustände werden diskret codiert. Je nach Umgebung bestehen sie aus Positionsinformationen oder zusätzlichen Merkmalen wie dem Transportstatus eines Containers.

## Deep Q-Learning (DQN)

Deep Q-Learning approximiert die Q-Funktion mit einem neuronalen Netzwerk. Damit können auch größere oder kontinuierliche Zustandsräume behandelt werden.

### Hauptkomponenten

1. **Q-Network**: Vorwärtspass zur Schätzung der Q-Werte
2. **Target Network**: Stabilisierte Zielwerte durch verzögertes Update
3. **Experience Replay**: Training auf zufälligen Mini-Batches aus einer Replay-Memory

### Lernprozess

Der Trainingsprozess basiert auf der Minimierung des Fehlers zwischen dem geschätzten Q-Wert und einem Zielwert:

```python
target_q = reward + γ × max(target_network(next_state))
loss = MSE(q_network(state)[action], target_q)
```

Die Netzwerkarchitektur kann flexibel angepasst werden und besteht typischerweise aus mehreren vollständig verbundenen Schichten mit ReLU-Aktivierung.

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
- Erweiterte Aufgaben mit Objekten wie Containern

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
- Ein Abbruchkriterium wurde erfüllt (z. B. zu viele Schleifen)
- Ein Hindernis wurde getroffen
- Die maximale Schrittanzahl wurde überschritten

## Szenarien

Zur Evaluation werden verschiedene Umgebungsmodi verwendet. Diese variieren beispielsweise durch:
- Fixe oder zufällige Start- und Zielpositionen
- Platzierung von Hindernissen
- Erweiterte Aufgaben wie Objekthandling

Die Szenarien lassen sich modular anpassen und sind zentral konfigurierbar.
