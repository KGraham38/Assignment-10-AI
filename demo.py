"""
Demo of DQN agent on LunarLander-v3 using the best performing neural network configuration.

Run after training and evaluation:
    python train.py
    python evaluate.py
"""

import torch.nn as nn
from agent import DQNAgent
from config import Config
from utils import get_device, make_env

def run_best(model_path: str, config: Config, episodes: int = 5):
    device = get_device()
    env = make_env(config.env_id, render_mode="human")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, config, device)
    agent.load(model_path)
    agent.epsilon = 0.0

    state, _ = env.reset()

    for episode in range(1, episodes + 1):
        # No fixed seed for stochastic behavior
        state, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Evaluation Episode {episode}: reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    # Initialize best performing config
    best_cfg = Config(
        experiment_name="wide",
        hidden_layers=(256, 256),
        activation_f=nn.ReLU,
        dropout_rate=0.0
    )

    # Run visual demo
    run_best("models/wide.pt", best_cfg)

