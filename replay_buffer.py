# Import deque for fixed-size replay storage
from collections import deque
# Import random sampling utilities
import random
# Import NumPy for array conversion
import numpy as np


# Define the replay buffer class
class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    # Initialize the buffer with a maximum capacity
    def __init__(self, capacity: int):
        # Store transitions in a deque that automatically removes the oldest items when full
        self.buffer = deque(maxlen=capacity)

    # Store one experience tuple in the buffer
    def push(self, state, action, reward, next_state, done) -> None:
        # Append the transition as a tuple with consistent data types
        self.buffer.append(
            (
                # Store the current state as a float32 NumPy array
                np.array(state, dtype=np.float32),
                # Store the action as an integer
                int(action),
                # Store the reward as a float
                float(reward),
                # Store the next state as a float32 NumPy array
                np.array(next_state, dtype=np.float32),
                # Store done as a float so it can be used directly in Bellman target math later
                float(done),
            )
        )

    # Randomly sample a batch of transitions from memory
    def sample(self, batch_size: int):
        # Randomly choose batch_size transitions from the buffer
        batch = random.sample(self.buffer, batch_size)
        # Unzip the batch into separate tuples of states, actions, rewards, next_states, and dones
        states, actions, rewards, next_states, dones = zip(*batch)

        # Return each piece as a NumPy array with a consistent dtype
        return (
            # Batch of states
            np.array(states, dtype=np.float32),
            # Batch of actions
            np.array(actions, dtype=np.int64),
            # Batch of rewards
            np.array(rewards, dtype=np.float32),
            # Batch of next states
            np.array(next_states, dtype=np.float32),
            # Batch of done flags
            np.array(dones, dtype=np.float32),
        )

    # Return the current number of stored transitions
    def __len__(self) -> int:
        # Let len(replay_buffer) return the size of the underlying deque
        return len(self.buffer)