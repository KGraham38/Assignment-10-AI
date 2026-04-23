import random

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from dqn_network import DQNNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """Implements action selection, experience storage, and DQN learning."""

    def __init__(self, state_size: int, action_size: int, config: Config, device: torch.device):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = device

        # Initialize policy and target networks using configurable architectures
        self.policy_net = DQNNetwork(
            state_size,
            config.hidden_layers,
            action_size,
            activation_f=config.activation_f,
            dropout_rate=config.dropout_rate
        ).to(device)
        self.target_net = DQNNetwork(
            state_size,
            config.hidden_layers,
            action_size,
            activation_f=config.activation_f,
            dropout_rate=config.dropout_rate
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(config.replay_capacity)

        self.epsilon = config.epsilon_start
        self.learn_step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        if len(self.memory) < self.config.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            max_next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q_values = rewards_t + self.config.gamma * max_next_q_values * (1.0 - dones_t)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step_count += 1
        if self.learn_step_count % self.config.target_update_every == 0:
            self.update_target_network()

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)