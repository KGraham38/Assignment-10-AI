# Import json for saving metrics to a JSON file
import json
# Import Path for simple file writing
from pathlib import Path

# Import matplotlib for plotting training progress
import matplotlib.pyplot as plt


# Define a class to track training rewards, losses, and moving averages
class TrainingLogger:
    """Tracks rewards and losses, saves metrics, and creates a plot."""

    # Initialize the logger with the size of the moving average window
    def __init__(self, moving_window: int = 100):
        # Store the number of recent episodes used in the moving average
        self.moving_window = moving_window
        # Store the reward from each episode
        self.episode_rewards = []
        # Store the moving average reward after each episode
        self.moving_averages = []
        # Store loss values observed during training
        self.losses = []

    # Add one completed episode's reward and most recent loss
    def add_episode(self, reward: float, loss: float | None) -> float:
        # Save the episode reward as a float
        self.episode_rewards.append(float(reward))
        # Only save the loss if a training step actually happened
        if loss is not None:
            self.losses.append(float(loss))

        # Get the most recent rewards within the moving average window
        recent = self.episode_rewards[-self.moving_window :]
        # Compute the moving average reward over the recent window
        moving_avg = sum(recent) / len(recent)
        # Save the moving average value
        self.moving_averages.append(float(moving_avg))
        # Return the moving average for immediate reporting
        return float(moving_avg)

    # Save all tracked metrics to a JSON file
    def save_metrics(self, path: str) -> None:
        # Package the tracked values into a dictionary
        data = {
            "episode_rewards": self.episode_rewards,
            "moving_averages": self.moving_averages,
            "losses": self.losses,
        }
        # Write the metrics dictionary to disk as formatted JSON
        Path(path).write_text(json.dumps(data, indent=2))

    # Create and save a training progress plot
    def plot(self, output_path: str) -> None:
        # Create a new figure with a fixed size
        plt.figure(figsize=(10, 5))
        # Plot raw episode rewards
        plt.plot(self.episode_rewards, label="Episode Reward")
        # Plot moving average rewards
        plt.plot(self.moving_averages, label=f"Moving Avg ({self.moving_window})")
        # Label the x-axis
        plt.xlabel("Episode")
        # Label the y-axis
        plt.ylabel("Reward")
        # Add a title to the plot
        plt.title("DQN Lunar Lander Training")
        # Show the legend
        plt.legend()
        # Adjust layout so labels fit cleanly
        plt.tight_layout()
        # Save the plot image to disk
        plt.savefig(output_path)
        # Close the figure to free memory
        plt.close()