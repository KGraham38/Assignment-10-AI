from pathlib import Path
import json
import tempfile

import numpy as np
import torch

from replay_buffer import ReplayBuffer
from agent import DQNAgent
from config import Config
from logger import TrainingLogger
from utils import get_device, set_seed, make_env


def test_replay_buffer():
    print("Running replay buffer test...")
    buffer = ReplayBuffer(capacity=10)

    state = np.zeros(8, dtype=np.float32)
    next_state = np.ones(8, dtype=np.float32)

    for i in range(6):
        buffer.push(state + i, i % 4, float(i), next_state + i, i % 2 == 0)

    assert len(buffer) == 6, f"Expected buffer length 6, got {len(buffer)}"

    states, actions, rewards, next_states, dones = buffer.sample(4)

    assert states.shape == (4, 8), f"Unexpected states shape: {states.shape}"
    assert actions.shape == (4,), f"Unexpected actions shape: {actions.shape}"
    assert rewards.shape == (4,), f"Unexpected rewards shape: {rewards.shape}"
    assert next_states.shape == (4, 8), f"Unexpected next_states shape: {next_states.shape}"
    assert dones.shape == (4,), f"Unexpected dones shape: {dones.shape}"

    print("PASS: replay buffer works\n")


def test_target_network_update():
    print("Running target network update test...")
    device = torch.device("cpu")
    config = Config(
        batch_size=4,
        min_replay_size=4,
        replay_capacity=100,
        target_update_every=2,
    )

    agent = DQNAgent(state_size=8, action_size=4, config=config, device=device)

    for param in agent.policy_net.parameters():
        with torch.no_grad():
            param.add_(1.0)

    different_before = False
    for p_policy, p_target in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        if not torch.allclose(p_policy, p_target):
            different_before = True
            break

    assert different_before, "Policy and target networks should differ before update"

    agent.update_target_network()

    for p_policy, p_target in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p_policy, p_target), "Target network did not sync correctly"

    print("PASS: target network update works\n")


def test_agent_learn():
    print("Running agent learn test...")
    device = torch.device("cpu")
    config = Config(
        batch_size=4,
        min_replay_size=4,
        replay_capacity=100,
        target_update_every=2,
    )

    agent = DQNAgent(state_size=8, action_size=4, config=config, device=device)

    for i in range(8):
        state = np.full(8, i, dtype=np.float32)
        next_state = np.full(8, i + 1, dtype=np.float32)
        action = i % 4
        reward = float(i)
        done = float(i % 2 == 0)
        agent.store_transition(state, action, reward, next_state, done)

    loss = agent.learn()
    assert loss is not None, "Expected a loss value from learn()"
    assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"

    print(f"PASS: agent learn works, loss = {loss:.6f}\n")


def test_logger():
    print("Running logger test...")
    logger = TrainingLogger(moving_window=3)

    logger.add_episode(10, 0.5)
    logger.add_episode(20, 0.4)
    logger.add_episode(30, 0.3)

    assert len(logger.episode_rewards) == 3
    assert len(logger.moving_averages) == 3
    assert len(logger.losses) == 3
    assert abs(logger.moving_averages[-1] - 20.0) < 1e-6

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = Path(tmpdir) / "metrics.json"
        plot_path = Path(tmpdir) / "plot.png"

        logger.save_metrics(str(metrics_path))
        logger.plot(str(plot_path))

        assert metrics_path.exists(), "Metrics file was not created"
        assert plot_path.exists(), "Plot file was not created"

        data = json.loads(metrics_path.read_text())
        assert "episode_rewards" in data
        assert "moving_averages" in data
        assert "losses" in data

    print("PASS: logger works\n")


def test_utils():
    print("Running utils test...")
    set_seed(42)

    device = get_device()
    assert str(device) in {"cpu", "cuda", "mps"}, f"Unexpected device: {device}"

    env = make_env("CartPole-v1")
    state, info = env.reset(seed=42)

    assert state is not None, "Environment reset returned no state"
    assert env.action_space.n == 2, "CartPole action space should be 2"

    env.close()
    print("PASS: utils work\n")


def main():
    tests = [
        test_replay_buffer,
        test_target_network_update,
        test_agent_learn,
        test_logger,
        test_utils,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {test.__name__}")
            print(f"Reason: {e}\n")

    print("=" * 40)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print("=" * 40)


if __name__ == "__main__":
    main()