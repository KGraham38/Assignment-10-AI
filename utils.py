# Import Python's random module for reproducible randomness
import random
# Import Optional type hint for optional render_mode argument
from typing import Optional

# Import NumPy for reproducible numerical randomness
import numpy as np
# Import PyTorch
import torch
# Import Gymnasium for environment creation
import gymnasium as gym


# Set seeds across Python, NumPy, and PyTorch for reproducibility
def set_seed(seed: int) -> None:
    # Seed Python's built-in random module
    random.seed(seed)
    # Seed NumPy random number generation
    np.random.seed(seed)
    # Seed PyTorch CPU random number generation
    torch.manual_seed(seed)
    # If CUDA is available, also seed all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Return the best available compute device
def get_device() -> torch.device:
    # Prefer CUDA if an NVIDIA GPU is available
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Otherwise use Apple's Metal backend if available
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # Fall back to CPU if no GPU backend is available
    return torch.device("cpu")


# Create and return a Gymnasium environment
def make_env(env_id: str, render_mode: Optional[str] = None):
    # Build the environment using the given environment id and optional render mode
    return gym.make(env_id, render_mode=render_mode)