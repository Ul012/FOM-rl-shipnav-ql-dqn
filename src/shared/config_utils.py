# src/shared/config_utils.py - Hilfsfunktionen für Konfiguration

from .config import *

def get_export_path(base_path: str) -> str:
    """
    Gibt den Exportpfad für das aktuelle Setup zurück (z. B. exports/v1).
    """
    full_path = os.path.join(base_path, SETUP_NAME)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def get_q_table_path(env_mode: str) -> str:
    """Q-Table Pfad basierend auf Modus"""
    return Q_TABLE_PATH_TEMPLATE.format(env_mode)


def get_dqn_model_path(env_mode: str) -> str:
    """DQN Model Pfad basierend auf Modus"""
    return DQN_MODEL_PATH_TEMPLATE.format(env_mode)


def get_q_learning_config() -> dict:
    """Gibt Q-Learning-spezifische Konfiguration zurück"""
    config = {
        'alpha': QL_ALPHA,
        'gamma': GAMMA,  # Gemeinsamer Parameter
        'episodes': EPISODES,  # Gemeinsamer Parameter
        'max_steps': MAX_STEPS,  # Gemeinsamer Parameter
        'seed': SEED  # Gemeinsamer Parameter
    }

    # Epsilon-Konfiguration abhängig von USE_EPSILON_DECAY
    if USE_EPSILON_DECAY:
        config.update({
            'epsilon_start': EPSILON_START,
            'epsilon_end': EPSILON_END,
            'epsilon_decay': EPSILON_DECAY,
            'use_epsilon_decay': True
        })
    else:
        config.update({
            'epsilon': QL_EPSILON_FIXED,
            'use_epsilon_decay': False
        })

    return config


def get_dqn_config() -> dict:
    """Gibt DQN-spezifische Konfiguration zurück"""
    config = {
        'state_size': DQN_STATE_SIZE,
        'action_size': N_ACTIONS,
        'learning_rate': DQN_LEARNING_RATE,
        'discount_factor': GAMMA,
        'buffer_size': DQN_BUFFER_SIZE,
        'batch_size': DQN_BATCH_SIZE,
        'target_update_freq': DQN_TARGET_UPDATE_FREQ,
        'hidden_size': DQN_HIDDEN_SIZE,
        'seed': SEED
    }

    # Epsilon-Konfiguration abhängig von USE_EPSILON_DECAY
    if USE_EPSILON_DECAY:
        config.update({
            'exploration_rate': EPSILON_START,
            'exploration_decay': EPSILON_DECAY,
            'min_exploration_rate': EPSILON_END,
            'use_epsilon_decay': True
        })
    else:
        config.update({
            'exploration_rate': DQN_EPSILON_FIXED,
            'use_epsilon_decay': False
        })

    return config


def get_training_config() -> dict:
    """Gibt gemeinsame Training-Konfiguration zurück"""
    return {
        'episodes': EPISODES,  # Gemeinsam
        'max_steps': MAX_STEPS,  # Gemeinsam
        'eval_episodes': EVAL_EPISODES,  # Gemeinsam
        'eval_max_steps': EVAL_MAX_STEPS,  # Gemeinsam
        'env_mode': ENV_MODE,  # Gemeinsam
        'seed': SEED,  # Gemeinsam
        'gamma': GAMMA,  # Gemeinsam
        'use_epsilon_decay': USE_EPSILON_DECAY,  # Gemeinsam
        'epsilon_start': EPSILON_START,  # Gemeinsam
        'epsilon_end': EPSILON_END,  # Gemeinsam
        'epsilon_decay': EPSILON_DECAY  # Gemeinsam
    }


def get_shared_config() -> dict:
    """Gibt alle geteilten Parameter zurück - für Vergleichsskripte"""
    return {
        'seed': SEED,
        'gamma': GAMMA,
        'episodes': EPISODES,
        'max_steps': MAX_STEPS,
        'eval_episodes': EVAL_EPISODES,
        'eval_max_steps': EVAL_MAX_STEPS,
        'env_mode': ENV_MODE,
        'grid_size': GRID_SIZE,
        'actions': ACTIONS,
        'n_actions': N_ACTIONS,
        'rewards': REWARDS,
        'use_epsilon_decay': USE_EPSILON_DECAY,
        'epsilon_start': EPSILON_START,
        'epsilon_end': EPSILON_END,
        'epsilon_decay': EPSILON_DECAY
    }


def validate_config() -> bool:
    """Validiert die Konfiguration und gibt Warnungen aus"""
    print("KONFIGURATION VALIDIERUNG")
    print("=" * 50)

    print("GEMEINSAME PARAMETER:")
    print(f"  Seed: {SEED}")
    print(f"  Gamma (Discount Factor): {GAMMA}")
    print(f"  Episodes: {EPISODES}")
    print(f"  Max Steps: {MAX_STEPS}")
    print(f"  Eval Episodes: {EVAL_EPISODES}")
    print(f"  Use Epsilon Decay: {USE_EPSILON_DECAY}")

    print("\nQ-LEARNING PARAMETER:")
    print(f"  Alpha (Learning Rate): {QL_ALPHA}")
    if USE_EPSILON_DECAY:
        print(f"  Epsilon Start: {EPSILON_START}")
        print(f"  Epsilon End: {EPSILON_END}")
        print(f"  Epsilon Decay: {EPSILON_DECAY}")
    else:
        print(f"  Epsilon (fest): {QL_EPSILON_FIXED}")

    print("\nDQN PARAMETER:")
    print(f"  Learning Rate: {DQN_LEARNING_RATE}")
    if USE_EPSILON_DECAY:
        print(f"  Epsilon Start: {EPSILON_START}")
        print(f"  Epsilon End: {EPSILON_END}")
        print(f"  Epsilon Decay: {EPSILON_DECAY}")
    else:
        print(f"  Epsilon (fest): {DQN_EPSILON_FIXED}")

    # Konsistenz-Checks
    warnings = []

    # Check: Beide verwenden gleiches Gamma
    dqn_config = get_dqn_config()
    if dqn_config['discount_factor'] != GAMMA:
        warnings.append("DQN verwendet anderes Gamma als Q-Learning!")

    # Check: Vernünftige Parameterbereiche
    if not (0.05 <= QL_ALPHA <= 0.5):
        warnings.append(f"Q-Learning Alpha außerhalb Standardbereich: {QL_ALPHA}")

    if not (0.0005 <= DQN_LEARNING_RATE <= 0.015):
        warnings.append(f"DQN Learning Rate außerhalb Standardbereich: {DQN_LEARNING_RATE}")

    # Check: Feste Epsilon-Werte sind gleich (für Fairness)
    if not USE_EPSILON_DECAY and QL_EPSILON_FIXED != DQN_EPSILON_FIXED:
        warnings.append(f"Unterschiedliche feste Epsilon-Werte: QL={QL_EPSILON_FIXED}, DQN={DQN_EPSILON_FIXED}")

    if warnings:
        print(f"\nWARNUNGEN:")
        for warning in warnings:
            print(f"  - {warning}")
        return False
    else:
        print(f"\nKonfiguration ist konsistent und fair!")
        return True

    print("=" * 50)


def print_config_summary():
    """Gibt eine übersichtliche Zusammenfassung der Konfiguration aus"""
    print("\nKONFIGURATION ZUSAMMENFASSUNG")
    print("=" * 60)

    shared = get_shared_config()
    ql = get_q_learning_config()
    dqn = get_dqn_config()

    print("GEMEINSAME PARAMETER (beide Algorithmen):")
    for key, value in shared.items():
        print(f"  {key}: {value}")

    print("\nQ-LEARNING SPEZIFISCH:")
    ql_specific = {k: v for k, v in ql.items() if
                   k not in shared or k in ['alpha', 'epsilon', 'epsilon_start', 'epsilon_end', 'epsilon_decay']}
    for key, value in ql_specific.items():
        print(f"  {key}: {value}")

    print("\nDQN SPEZIFISCH:")
    dqn_specific = {k: v for k, v in dqn.items()
                    if k not in shared and k not in ['discount_factor', 'action_size', 'exploration_rate',
                                                     'exploration_decay', 'min_exploration_rate']}
    for key, value in dqn_specific.items():
        print(f"  {key}: {value}")

    print("=" * 60)

def prepare_export_dirs():
    """
    Erstellt Exportverzeichnisse für alle relevanten SETUPs inkl. 'combined'.
    """
    for _path in [EXPORT_PATH_QL, EXPORT_PATH_DQN, EXPORT_PATH_COMP]:
        full_path = os.path.join(_path, SETUP_NAME)
        os.makedirs(full_path, exist_ok=True)
        os.makedirs(os.path.join(full_path, "combined"), exist_ok=True)


if __name__ == "__main__":
    # Führe Validierung aus wenn config_utils.py direkt ausgeführt wird
    validate_config()
    print_config_summary()