"""Tests for device detection utility."""
import torch
import pytest
from vidbot_utils.device import get_device


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_returns_valid_type():
    device = get_device()
    assert device.type in ("cuda", "mps", "cpu")


def test_get_device_mps_on_apple_silicon():
    """On Apple Silicon Mac, MPS should be available and selected."""
    device = get_device()
    if torch.backends.mps.is_available():
        assert device.type == "mps"


def test_tensor_creation_on_device():
    device = get_device()
    t = torch.randn(4, 4, device=device)
    assert t.device.type == device.type


def test_tensor_to_device():
    device = get_device()
    t = torch.randn(3, 3)
    t_dev = t.to(device)
    assert t_dev.device.type == device.type


def test_tensor_operations_on_device():
    device = get_device()
    a = torch.randn(3, 3, device=device)
    b = torch.randn(3, 3, device=device)
    c = a @ b
    assert c.device.type == device.type
    assert c.shape == (3, 3)
