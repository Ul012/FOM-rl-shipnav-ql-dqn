# 📚 Produkt-Dokumentation und Entwicklungsstruktur

## Lokale Dokumentation

```bash
# Start der Dokumentation im Live-Modus
mkdocs serve
# Erreichbar unter: http://127.0.0.1:8000
```

```bash
# Generierung statischer HTML-Dateien
mkdocs build
```

```bash
# Dokumentation mit Auto-Reload auf anderem Port
mkdocs serve --dev-addr=127.0.0.1:8001
```

## Struktur der Dokumentation

```plaintext
docs/
├── index.md              # Projektübersicht und Algorithmusvergleich
├── setup.md              # Installation und Projektstart
├── funktionsweise.md     # Algorithmische Grundlagen
├── training.md           # Training und Konfiguration
├── visualisierung.md     # Evaluation und Analyse
└── dokumentation.md      # Entwickler- und Wartungsdokumentation
```

## Modularer Aufbau

Das Projekt ist in separaten Modulen strukturiert:

- `q_learning/` und `dqn/`: eigenständige Algorithmenbereiche
- `shared/`: Konfiguration, Umgebungen, Utilities
- `comparison/`: Scripts zur Visualisierung und Evaluation

## Dokumentationssystem

Die `.md`-Dateien in `docs/` sind mit [MkDocs](https://www.mkdocs.org/) kompatibel und dienen der projektnahen Beschreibung für Entwicklung, Analyse und Präsentation.

## Erweiterbarkeit

- Neue Szenarien lassen sich durch Ergänzen in der `SCENARIOS`-Struktur einfügen
- Vergleichsmetriken und Visualisierungen sind unabhängig von der Trainingsmethode
- Modelle und Q-Tabellen werden szenariobasiert gespeichert

## Pflege und Wartung

- Dokumentation synchron zur Codebasis halten
- Versionskontrolle für Änderungen nutzen
- Einheitliche Begriffe und Formatierung sicherstellen

Die Dokumentation dient als projektnaher Referenzrahmen und sollte fortlaufend mit dem Code synchronisiert werden.
