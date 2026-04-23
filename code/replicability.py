"""Helpers for deterministic experiment setup."""

import random

import numpy as np
import torch


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    print(f"Set SEED: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_seed_fn(seed):
    """Create a DataLoader worker initializer with deterministic offsets."""

    def seed_worker(worker_id):
        """Seed one worker process using the global seed plus worker index."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def get_generator(seed):
    """Create a seeded PyTorch generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
