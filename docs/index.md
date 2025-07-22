# 📘 Projektübersicht: Q-Learning vs Deep Q-Learning

Dieses Projekt untersucht die Anwendung von Reinforcement Learning zur Navigation autonomer Agenten in Gitterumgebungen. Es werden zwei Ansätze implementiert und verglichen:

- **Q-Learning** – klassisches tabellenbasiertes Verfahren
- **Deep Q-Learning (DQN)** – Approximation über neuronale Netze

Ziel ist ein reproduzierbarer Vergleich in mehreren Szenarien mit variabler Komplexität.

---

## 🔍 Dokumentationsstruktur

| Thema               | Beschreibung |
|--------------------|--------------|
| [⚙️ Setup](setup.md) | Einrichtung, Abhängigkeiten, Projektstart |
| [🧠 Funktionsweise](funktionsweise.md) | Technische Grundlagen beider Algorithmen |
| [🎯 Training](training.md) | Trainingsdurchführung und Konfigurationsmöglichkeiten |
| [📊 Visualisierung](visualisierung.md) | Darstellung und Vergleich von Ergebnissen |
| [📚 Entwicklung](dokumentation.md) | Hinweise zur Projektstruktur und Erweiterung |

---

## 🗺️ Unterstützte Szenarien

- **Static**: Feste Start-/Zielposition
- **Random Start**: Zufälliger Startpunkt
- **Random Goal**: Zufälliges Ziel
- **Random Obstacles**: Zufällige Hindernisse
- **Container**: Aufgabe mit Pickup und Dropoff

---

## 🧪 Vergleichsoptionen

Das Projekt enthält drei Varianten zur algorithmischen Ergebnisvisualisierung. Die zentrale Auswertung erfolgt über `1_compare_algorithms_overview.py`. Auf Basis der erzeugten CSV können weitere Visualisierungs-Varianten erzeugt werden.