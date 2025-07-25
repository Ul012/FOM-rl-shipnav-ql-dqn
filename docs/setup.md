# Installation und Setup

## Voraussetzungen

Für das Projekt wird Python 3.8 oder höher benötigt. Empfohlen wird die Nutzung einer virtuellen Umgebung.
Das Projekt wurde mit Python 3.11.5 entwickelt und getestet.
 Die Abhängigkeiten sind in der Datei `requirements.txt` definiert.

### Hardware-Unterstützung

- **CPU:** Alle Algorithmen funktionieren auf Standard-CPUs
- **GPU (optional):** DQN unterstützt CUDA-beschleunigtes Training bei verfügbarer NVIDIA-GPU
- **RAM:** Mindestens 4GB empfohlen für größere Replay-Buffer

## Einrichtung

```bash
# Repository klonen
git clone https://github.com/DeinUser/ship-navigation-ql-dqn.git
cd ship-navigation-ql-dqn

# Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate    # Linux/Mac

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Schnelltest

Zur Überprüfung der Installation kann ein einfacher Importtest durchgeführt werden:

```bash
cd src
python -c "from shared.envs import GridEnvironment; print('Setup erfolgreich')"
```

## Strukturüberblick

Die zentralen Komponenten befinden sich im Ordner `src/`. Für beide Algorithmen gibt es getrennte Trainings- und Evaluationsskripte. Die Dokumentation ist in `docs/` abgelegt und mit MkDocs aufrufbar.

Ergebnisse wie Lernkurven, Erfolgsraten und Vergleichsplots werden automatisch im Verzeichnis `exports/` gespeichert. Für jedes Szenario und jeden Algorithmus wird ein separater Export erzeugt.
