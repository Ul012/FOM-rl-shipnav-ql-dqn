# visualize_policy.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Drittanbieter
import pygame
import time

# Lokale Module
from src.shared.config import (ENV_MODE, MAX_STEPS, CELL_SIZE, FRAME_DELAY, EXPORT_PDF, EXPORT_PATH, REWARDS)
from shared.envs.grid_environment import GridEnvironment
from shared.envs.container_environment import ContainerShipEnv

# Utils
from utils.common import set_all_seeds, obs_to_state, setup_export
from utils.qlearning import load_q_table, get_best_action
from utils.position import get_position


# ============================================================================
# Visualisierung
# ============================================================================

# Darstellung des Grids mit Agent und Policy
def draw_grid(screen, font, env, agent_pos, Q):
    colors = {
        'background': (224, 247, 255),
        'grid_line': (200, 200, 200),
        'text': (0, 0, 0)
    }

    actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    screen.fill(colors['background'])

    # Grid zeichnen
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, colors['grid_line'], rect, 1)
            pos = (i, j)

            # Symbol bestimmen
            if pos == agent_pos:
                symbol = "🚢"
            elif hasattr(env, 'start_pos') and pos == env.start_pos:
                symbol = "🧭"
            elif hasattr(env, "pickup_pos") and pos == env.pickup_pos:
                symbol = "📦"
            elif hasattr(env, "dropoff_pos") and pos == env.dropoff_pos:
                symbol = "🏁"
            elif hasattr(env, "goal_pos") and pos == env.goal_pos:
                symbol = "🏁"
            elif pos in getattr(env, "obstacles", []):
                symbol = "🪨"
            else:
                # Policy-Pfeil
                if ENV_MODE == "container":
                    state = obs_to_state((i, j, 0), ENV_MODE, env.grid_size)
                else:
                    state = env.pos_to_state((i, j))

                if state < Q.shape[0]:
                    action = get_best_action(Q, state)
                    symbol = actions_map[action]
                else:
                    symbol = "?"

            # Symbol zeichnen
            text_surface = font.render(symbol, True, colors['text'])
            text_rect = text_surface.get_rect(
                center=(j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2)
            )
            screen.blit(text_surface, text_rect)

    pygame.display.flip()


# Speicherung des Screenshots
def save_screenshot(screen):
    if EXPORT_PDF:
        screenshot_path = f"{EXPORT_PATH}/agent_final_position.png"
        pygame.image.save(screen, screenshot_path)
        print(f"Screenshot gespeichert: {screenshot_path}")


# ============================================================================
# Hauptfunktion
# ============================================================================

# Ausführung des Agenten mit gelernter Policy
def run_agent():
    # Seed für Reproduzierbarkeit setzen
    set_all_seeds()

    print(f"Starte Visualisierung ({ENV_MODE}-Modus)...")

    # Initialisierung
    env = ContainerShipEnv() if ENV_MODE == "container" else GridEnvironment(mode=ENV_MODE)
    Q = load_q_table(ENV_MODE)

    if Q is None:
        print(f"FEHLER: Q-Tabelle nicht gefunden: {Q_TABLE_PATH}")
        print("Bitte führen Sie zuerst das Training aus.")
        sys.exit(1)

    setup_export()

    # Pygame
    pygame.init()
    screen_size = CELL_SIZE * env.grid_size
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption(f"Agent Policy - {ENV_MODE}")
    font = pygame.font.SysFont("Segoe UI Emoji", 40)

    # Episode starten
    obs, _ = env.reset()
    agent_pos = get_position(obs, ENV_MODE)

    step_count = 0
    total_reward = 0

    print(f"Start: {agent_pos}")

    # Ersten Frame zeichnen
    draw_grid(screen, font, env, agent_pos, Q)
    time.sleep(1.0)

    # Hauptschleife
    running = True
    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        # Aktion ausführen
        state = obs_to_state(obs, ENV_MODE, env.grid_size)
        action = get_best_action(Q, state)
        obs, reward, terminated, truncated, _ = env.step(action)

        # Update
        agent_pos = get_position(obs, ENV_MODE)
        step_count += 1
        total_reward += reward

        print(f"Schritt {step_count}: {agent_pos}, Reward: {reward}")

        # Zeichnen
        draw_grid(screen, font, env, agent_pos, Q)
        time.sleep(FRAME_DELAY)

        # Ende prüfen
        if terminated or truncated or step_count >= MAX_STEPS:
            success = (ENV_MODE == "container" and reward == REWARDS["dropoff"]) or \
                      (ENV_MODE != "container" and reward == REWARDS["goal"])

            print(f"\nEpisode beendet nach {step_count} Schritten")
            print(f"Ziel erreicht: {'Ja' if success else 'Nein'}")
            print(f"Gesamt-Reward: {total_reward}")

            # Screenshot speichern
            save_screenshot(screen)
            time.sleep(2.0)
            running = False

    pygame.quit()


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    run_agent()