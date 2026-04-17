import json
from pathlib import Path

import matplotlib.pyplot as plt


class TrainingLogger:
    """Tracks rewards and losses, saves metrics, and creates a plot."""

    def __init__(self, moving_window: int = 100):
        self.moving_window = moving_window
        self.episode_rewards = []
        self.moving_averages = []
        self.losses = []

    def add_episode(self, reward: float, loss: float | None) -> float:
        self.episode_rewards.append(float(reward))
        if loss is not None:
            self.losses.append(float(loss))

        recent = self.episode_rewards[-self.moving_window :]
        moving_avg = sum(recent) / len(recent)
        self.moving_averages.append(float(moving_avg))
        return float(moving_avg)

    def save_metrics(self, path: str) -> None:
        data = {
            "episode_rewards": self.episode_rewards,
            "moving_averages": self.moving_averages,
            "losses": self.losses,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def plot(self, output_path: str) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.plot(self.moving_averages, label=f"Moving Avg ({self.moving_window})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN Lunar Lander Training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()