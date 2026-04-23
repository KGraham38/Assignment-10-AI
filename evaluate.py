"""
Evaluate a trained DQN agent across multiple neural network configurations.

Run after training:
    python evaluate.py
"""

import os
import json
import numpy as np
import torch.nn as nn
from agent import DQNAgent
from config import Config
from utils import get_device, make_env


# Output directory
METRICS_DIR = "metrics"

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

def print_eval_metrics(rewards, config):
    '''
    Calculate and print evaluation metrics to console
    '''
    # Print header
    print("\n" + "-" * 40 )
    print(" Evaluation Metrics ".center(40))
    print("-" * 40)

    print(f"Model: {config.experiment_name}")

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
    print("\n")

def evaluate(model_path: str, config: Config, episodes: int = 5, render_mode: str | None = "Human"):
    device = get_device()
    env = make_env(config.env_id, render_mode=render_mode)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, config, device)
    agent.load(model_path)
    agent.epsilon = 0.0

    rewards = []

    # Printing model header
    model_name = model_path.split("/")[-1].replace(".pt", "")
    print("\n" + "=" * 40)
    print(f"Evaluating: {model_name}".center(40))
    print("=" * 40)

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

    return rewards


if __name__ == "__main__":
    configs = {
        "baseline": Config(
            experiment_name="baseline",
            hidden_layers=(128, 128),
            activation_f=nn.ReLU,
            dropout_rate=0.0),
        "wide": Config(
            experiment_name="wide",
            hidden_layers=(256, 256),
            activation_f=nn.ReLU,
            dropout_rate=0.0),
        "deep": Config(
            experiment_name="deep",
            hidden_layers=(128, 128, 128),
            activation_f=nn.ReLU,
            dropout_rate=0.0),
        "dropout": Config(
            experiment_name="dropout",
            hidden_layers=(128, 128),
            activation_f=nn.ReLU,
            dropout_rate=0.2)
    }
    models = ["baseline", "wide", "deep", "dropout"]
    results = {}

    # Evaluate each model config
    # Running in non-rendering mode for faster evaluation
    for name in models:
        cfg = configs[name]

        rewards = evaluate(f"models/{name}.pt", config=cfg, episodes=10, render_mode="None")

        # Save and print evaluation metrics
        save_eval_metrics(rewards, cfg, f"metrics/eval_metrics_{name}.json")
        print_eval_metrics(rewards, cfg)

        # Store summarized results for ranking
        rewards_np = np.array(rewards)
        results[name] = {
            "mean": float(np.mean(rewards_np)),
            "std": float(np.std(rewards_np)),
            "success": float(np.mean(rewards_np >= cfg.solve_score)),
            "cv": float(np.std(rewards_np) / np.mean(rewards_np)) if np.mean(rewards_np) != 0 else 0.0
        }

    # Find the best model based on mean reward
    best_name, best_stats = max(results.items(), key=lambda x: x[1]["mean"])

    # Print best model summary
    print("\n" + "=" * 40)
    print(" Best Performing Model ".center(40))
    print("=" * 40)
    print(f"Model: {best_name}")
    print(f"Mean Reward: {best_stats['mean']:.2f} ± {best_stats['std']:.2f}")
    print(f"Success Rate: {best_stats['success'] * 100:.1f}%")
    print(f"CV: {best_stats['cv']:.3f}")
    print("-" * 40)