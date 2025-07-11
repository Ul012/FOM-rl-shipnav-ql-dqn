# src/shared/config_utils.py - Hilfsfunktionen für Konfiguration

from .config import *


def get_q_table_path(env_mode: str) -> str:
    """Q-Table Pfad basierend auf Modus"""
    return Q_TABLE_PATH_TEMPLATE.format(env_mode)


def get_dqn_model_path(env_mode: str) -> str:
    """DQN Model Pfad basierend auf Modus"""
    return DQN_MODEL_PATH_TEMPLATE.format(env_mode)


def get_q_learning_config() -> dict:
    """Gibt Q-Learning-spezifische Konfiguration zurück"""
    return {
        'alpha': QL_ALPHA,
        'epsilon': QL_EPSILON,
        'gamma': GAMMA,  # Gemeinsamer Parameter
        'episodes': EPISODES,  # Gemeinsamer Parameter
        'max_steps': MAX_STEPS,  # Gemeinsamer Parameter
        'seed': SEED  # Gemeinsamer Parameter
    }


def get_dqn_config() -> dict:
    """Gibt DQN-spezifische Konfiguration zurück"""
    return {
        'state_size': DQN_STATE_SIZE,
        'action_size': N_ACTIONS,  # Gemeinsamer Parameter
        'learning_rate': DQN_LEARNING_RATE,
        'discount_factor': GAMMA,  # Gemeinsamer Parameter!
        'exploration_rate': DQN_EPSILON_START,
        'exploration_decay': DQN_EPSILON_DECAY,
        'min_exploration_rate': DQN_EPSILON_END,
        'buffer_size': DQN_BUFFER_SIZE,
        'batch_size': DQN_BATCH_SIZE,
        'target_update_freq': DQN_TARGET_UPDATE_FREQ,
        'hidden_size': DQN_HIDDEN_SIZE,
        'seed': SEED  # Gemeinsamer Parameter
    }


def get_training_config() -> dict:
    """Gibt gemeinsame Training-Konfiguration zurück"""
    return {
        'episodes': EPISODES,  # Gemeinsam
        'max_steps': MAX_STEPS,  # Gemeinsam
        'eval_episodes': EVAL_EPISODES,  # Gemeinsam
        'eval_max_steps': EVAL_MAX_STEPS,  # Gemeinsam
        'env_mode': ENV_MODE,  # Gemeinsam
        'seed': SEED,  # Gemeinsam
        'gamma': GAMMA  # Gemeinsam
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
        'rewards': REWARDS
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

    print("\nQ-LEARNING PARAMETER:")
    print(f"  Alpha (Learning Rate): {QL_ALPHA}")
    print(f"  Epsilon (Exploration): {QL_EPSILON}")

    print("\nDQN PARAMETER:")
    print(f"  Learning Rate: {DQN_LEARNING_RATE}")
    print(f"  Epsilon Start: {DQN_EPSILON_START}")
    print(f"  Epsilon End: {DQN_EPSILON_END}")
    print(f"  Epsilon Decay: {DQN_EPSILON_DECAY}")

    # Konsistenz-Checks
    warnings = []

    # Check: Beide verwenden gleiches Gamma
    dqn_config = get_dqn_config()
    if dqn_config['discount_factor'] != GAMMA:
        warnings.append("DQN verwendet anderes Gamma als Q-Learning!")

    # Check: Vernünftige Parameterbereiche
    if not (0.05 <= QL_ALPHA <= 0.5):
        warnings.append(f"Q-Learning Alpha außerhalb Standardbereich: {QL_ALPHA}")

    if not (0.0005 <= DQN_LEARNING_RATE <= 0.01):
        warnings.append(f"DQN Learning Rate außerhalb Standardbereich: {DQN_LEARNING_RATE}")

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
    ql_specific = {k: v for k, v in ql.items() if k not in shared or k in ['alpha', 'epsilon']}
    for key, value in ql_specific.items():
        print(f"  {key}: {value}")

    print("\nDQN SPEZIFISCH:")
    dqn_specific = {k: v for k, v in dqn.items()
                    if k not in shared and k not in ['discount_factor', 'action_size']}
    for key, value in dqn_specific.items():
        print(f"  {key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    # Führe Validierung aus wenn config_utils.py direkt ausgeführt wird
    validate_config()
    print_config_summary()