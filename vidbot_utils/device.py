"""
Device detection utility for cross-platform PyTorch support.
Supports: CUDA (NVIDIA GPU), MPS (Apple Silicon), CPU (fallback).

Optional override:
    VIDBOT_DEVICE=cpu|mps|cuda
"""

import os

import torch


def get_device() -> torch.device:
    """Return the best available torch device."""
    override = os.environ.get("VIDBOT_DEVICE", "").strip().lower()
    if override:
        if override == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("VIDBOT_DEVICE=cuda requested, but CUDA is not available.")
            return torch.device("cuda")
        if override == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("VIDBOT_DEVICE=mps requested, but MPS is not available.")
            return torch.device("mps")
        if override == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported VIDBOT_DEVICE override: {override}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
