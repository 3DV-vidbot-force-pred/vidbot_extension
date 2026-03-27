"""
Device detection utility for cross-platform PyTorch support.
Supports: CUDA (NVIDIA GPU), MPS (Apple Silicon), CPU (fallback).
"""

import torch


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
