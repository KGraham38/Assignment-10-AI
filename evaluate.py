"""
Evaluate a trained DQN model.

Run after training:
    python evaluate.py
"""

import json
import numpy as np
from agent import DQNAgent
from config import Config
from utils import get_device, make_env

def save_eval_metrics(rewards, config, path="eval_metrics.json"):
    '''
    Save evaluation metrics to a JSON file
    '''
    rewards = np.array(rewards)

    metrics = {
        "experiment": getattr(config, "experiment_name", "default"),
        "env_id": config.env_id,
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "success_rate": float(np.mean(rewards >= config.solve_score)),
        "cv": float(np.std(rewards) / np.mean(rewards)) if np.mean(rewards) != 0 else 0.0
    }

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

def log_eval_metrics(rewards, config):
    '''
    Calculate and print evaluation metrics to console
    '''
    # Print header
    print("\n" + "-" * 40 )
    print(" Evaluation Metrics ".center(40))
    print("-" * 40)

    # print(f"Experiment: {config.experiment_name}")

    rewards = np.array(rewards)
    n = len(rewards)

    mean = np.mean(rewards)
    std = np.std(rewards)
    min_r = np.min(rewards)
    max_r = np.max(rewards)

    success_rate = np.mean(rewards >= config.solve_score)
    cv = std / mean if mean != 0 else 0

    print(f"Environment: {config.env_id}")
    print(f"Episodes: {n}")

    print(f"Mean Reward: {mean:.2f} ± {std:.2f}")
    print(f"Min Reward: {min_r:.2f}")
    print(f"Max Reward: {max_r:.2f}")

    print(f"Success Rate (≥ {config.solve_score}): {success_rate * 100:.1f}%")
    print(f"Variability (CV): {cv:.2f}")

    print("-" * 40 )
    print(" Evaluation Complete ".center(40))
    print("-" * 40)

def evaluate(model_path: str, episodes: int = 5, render_mode: str | None = "Human"):
    config = Config()
    device = get_device()
    env = make_env(config.env_id, render_mode=render_mode)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, config, device)
    agent.load(model_path)
    agent.epsilon = 0.0

    rewards = []

    for episode in range(1, episodes + 1):
        state, info = env.reset(seed=config.seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        print(f"Evaluation Episode {episode}: reward = {total_reward:.2f}")

    env.close()

    # Modeling contribution: added additional statistical evaluation metrics
    log_eval_metrics(rewards, config)
    save_eval_metrics(rewards, config)

    return rewards


if __name__ == "__main__":
    cfg = Config()
    evaluate(cfg.model_path, episodes=10, render_mode="None")