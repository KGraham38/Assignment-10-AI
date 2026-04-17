from dataclasses import dataclass


@dataclass
class Config:
    env_id: str = "LunarLander-v3"
    seed: int = 42

    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 100_000
    min_replay_size: int = 5_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    target_update_every: int = 1_000
    train_every: int = 1

    hidden_size_1: int = 128
    hidden_size_2: int = 128

    max_episodes: int = 500
    max_steps_per_episode: int = 1_000

    solve_score: float = 200.0
    moving_average_window: int = 100

    model_path: str = "lunar_lander_dqn.pt"
    metrics_path: str = "training_metrics.json"
    plot_path: str = "training_plot.png"