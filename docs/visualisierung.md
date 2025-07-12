# Visualisierung und Analyse

## Automatische Diagrammerzeugung

Im Rahmen des Trainings werden automatisch visuelle Ausgaben erzeugt. Dazu gehören:

- **Lernkurven**: Entwicklung des Rewards über die Episoden
- **Erfolgsratenverlauf**: Anteil erfolgreicher Episoden über Zeit
- **Vergleichsplots**: Aggregierte Darstellung über mehrere Runs

Diese Diagramme werden pro Szenario gespeichert. Zusätzlich wird für jede Methode eine kombinierte Kurve über alle Szenarien erstellt (Ordner `exports/combined`).

## Vergleichsvisualisierungen

Zum algorithmischen Vergleich stehen drei Visualisierungsformen zur Verfügung:

1. **2x3-Vergleich** (`compare_algorithms_2x3.py`)  
   - Führt eine standardisierte Evaluation durch  
   - Exportiert CSV-Datei und Vergleichsplot  
   - Grundlage für weitere Visualisierungen

2. **2x2 V1** (`compare_algorithms_2x2_v1.py`)  
   - Darstellung in vier Panels  
   - Erfolgsrate, Reward, Schritte und Scatterplot zur Effizienz

3. **2x2 V2** (`compare_algorithms_2x2_v2.py`)  
   - Fokus auf Vergleich einzelner Szenarien  
   - Enthält Erfolgs-Heatmap  
   - Container-Szenario ist enthalten

## Exportformate

Die Ergebnisse werden als `.pdf` und `.csv` gespeichert. Je nach Konfiguration können zusätzlich agentenbasierte Visualisierungen erstellt werden (z. B. finale Positionen).

Die benutzerdefinierten Exporte befinden sich im jeweiligen `exports/`-Unterverzeichnis pro Algorithmus.
