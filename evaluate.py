"""
Evaluate a trained DQN model.

Run after training:
    python evaluate.py
"""

from agent import DQNAgent
from config import Config
from utils import get_device, make_env


def evaluate(model_path: str, episodes: int = 5, render_mode: str | None = "human"):
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
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average evaluation reward: {avg_reward:.2f}")
    return rewards


if __name__ == "__main__":
    cfg = Config()
    evaluate(cfg.model_path, episodes=3, render_mode="human")