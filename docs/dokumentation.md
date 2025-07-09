# Projektdokumentation und Entwicklung

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

## Softwarearchitektur

Das Projekt ist modular aufgebaut. Q-Learning und DQN sind in getrennten Verzeichnissen implementiert. Gemeinsame Komponenten wie Umgebungen und Konfigurationsparameter sind zentral in `shared/` abgelegt. Vergleichsskripte befinden sich in einem eigenen Modul.

Zentrale Konzepte:
- **Trennung der Verantwortlichkeiten** (Single Responsibility)
- **Modularität** für Wiederverwendbarkeit
- **Zentrale Konfiguration** über `shared/config.py`

## Dokumentationsworkflow

1. Änderungen an `.md`-Dateien in `docs/` vornehmen
2. Vorschau lokal mit `mkdocs serve`
3. Bei Bedarf statisches HTML erzeugen mit `mkdocs build`
4. Veröffentlichung optional über GitHub Pages

## Erweiterungen

Für die Dokumentation können bei Bedarf Plugins eingesetzt werden, z. B. zur Darstellung von Diagrammen oder zur PDF-Erzeugung. Diese sind in `requirements.txt` und `mkdocs.yml` konfigurierbar.

## Pflege und Wartung

- Dokumentation synchron zur Codebasis halten
- Versionskontrolle für Änderungen nutzen
- Einheitliche Begriffe und Formatierung sicherstellen

Die Dokumentation dient sowohl zur Einführung als auch zur technischen Referenz und sollte regelmäßig auf Aktualität geprüft werden.
