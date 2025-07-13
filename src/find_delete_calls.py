import os

# Verzeichnis, in dem gesucht wird (Passe den Pfad an dein Projekt an)
search_dir = "C:/Users/840171/OneDrive/Dokumente/Studium/Python/RL-Schiffsnavigation/ship-navigation-ql-dqn/src"

# Schlüsselwörter, nach denen gesucht wird
keywords = ["os.remove", "shutil.rmtree", ".unlink(", ".rmdir("]

print(f"Suche nach Löschbefehlen in {search_dir} ...\n")

for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if any(kw in line for kw in keywords):
                        print(f"{filepath} (Zeile {i}): {line.strip()}")

print("\nSuche abgeschlossen.")
