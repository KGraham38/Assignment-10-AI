"""
Train a DQN agent on LunarLander-v3 using multiple neural network configurations.

Run:
    python train.py
"""

import os
import torch.nn as nn
from agent import DQNAgent
from config import Config
from logger import TrainingLogger
from utils import get_device, make_env, set_seed


# Ensure output directories
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def train_dqn(config: Config):
    set_seed(config.seed)
    device = get_device()
    env = make_env(config.env_id)

    state, info = env.reset(seed=config.seed)
    env.action_space.seed(config.seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, config, device)
    logger = TrainingLogger(config.moving_average_window)

    total_steps = 0

    print(f"Using device: {device}")
    print(f"State size: {state_size}, Action size: {action_size}")

    for episode in range(1, config.max_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        last_loss = None

        for step in range(config.max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            if total_steps % config.train_every == 0:
                loss = agent.learn()
                if loss is not None:
                    last_loss = loss

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break

        agent.decay_epsilon()
        moving_avg = logger.add_episode(episode_reward, last_loss)

        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg({config.moving_average_window}): {moving_avg:8.2f} | "
            f"Epsilon: {agent.epsilon:6.3f} | "
            f"Loss: {last_loss if last_loss is not None else 'n/a'}"
        )

        if moving_avg >= config.solve_score and episode >= config.moving_average_window:
            print(f"\nSolved with moving average reward {moving_avg:.2f}.")
            break

    agent.save(f"models/{config.experiment_name}.pt")
    logger.save_metrics(f"metrics/train_metrics_{config.experiment_name}.json")
    logger.plot(f"plots/train_plot_{config.experiment_name}.png")
    env.close()

    print(f"\nModel saved to: models/{config.experiment_name}.pt")
    print(f"Metrics saved to: metrics/train_metrics_{config.experiment_name}.json")
    print(f"Plot saved to: plots/train_plot_{config.experiment_name}.png")

    return agent, logger


if __name__ == "__main__":
    # Neural network configurations to test
    experiments = {
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
    results = {}

    # Run DQN experiments across different model configs and log training results
    for name, cfg in experiments.items():
        print("\n" + "-" * 40)
        print(f"Experiment: {name}")
        print("-" * 40)

        agent, logger = train_dqn(cfg)
        results[name] = logger