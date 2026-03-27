"""Tests verifying PyTorch operations work correctly on the MPS backend.
These test common operations used throughout the codebase to ensure
Apple Silicon compatibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
from vidbot_utils.device import get_device

DEVICE = get_device()


class TestBasicTensorOps:
    def test_matmul(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 4, device=DEVICE)
        c = a @ b
        assert c.shape == (4, 4)
        assert c.device.type == DEVICE.type

    def test_batch_matmul(self):
        a = torch.randn(2, 4, 4, device=DEVICE)
        b = torch.randn(2, 4, 4, device=DEVICE)
        c = torch.bmm(a, b)
        assert c.shape == (2, 4, 4)

    def test_cross_product(self):
        a = torch.randn(2, 3, device=DEVICE)
        b = torch.randn(2, 3, device=DEVICE)
        c = torch.cross(a, b, dim=-1)
        assert c.shape == (2, 3)

    def test_eye_on_device(self):
        t = torch.eye(4, device=DEVICE)
        assert t.shape == (4, 4)
        assert t[0, 0] == 1.0
        assert t[0, 1] == 0.0

    def test_randn_on_device(self):
        t = torch.randn(3, 3, device=DEVICE)
        assert t.device.type == DEVICE.type

    def test_arange_on_device(self):
        t = torch.arange(10, device=DEVICE)
        assert t.shape == (10,)

    def test_meshgrid(self):
        xx, yy = torch.meshgrid(
            torch.arange(4, device=DEVICE),
            torch.arange(4, device=DEVICE),
            indexing="ij",
        )
        assert xx.shape == (4, 4)


class TestNNModules:
    def test_conv2d(self):
        conv = nn.Conv2d(3, 16, 3, padding=1).to(DEVICE)
        x = torch.randn(1, 3, 32, 32, device=DEVICE)
        y = conv(x)
        assert y.shape == (1, 16, 32, 32)

    def test_conv1d(self):
        conv = nn.Conv1d(64, 32, 3, padding=1).to(DEVICE)
        x = torch.randn(2, 64, 16, device=DEVICE)
        y = conv(x)
        assert y.shape == (2, 32, 16)

    def test_conv3d(self):
        conv = nn.Conv3d(1, 8, 3, padding=1).to(DEVICE)
        x = torch.randn(1, 1, 8, 8, 8, device=DEVICE)
        y = conv(x)
        assert y.shape == (1, 8, 8, 8, 8)

    def test_linear(self):
        linear = nn.Linear(64, 32).to(DEVICE)
        x = torch.randn(2, 64, device=DEVICE)
        y = linear(x)
        assert y.shape == (2, 32)

    def test_layer_norm(self):
        ln = nn.LayerNorm(64).to(DEVICE)
        x = torch.randn(2, 8, 64, device=DEVICE)
        y = ln(x)
        assert y.shape == (2, 8, 64)

    def test_group_norm(self):
        gn = nn.GroupNorm(4, 16).to(DEVICE)
        x = torch.randn(2, 16, 8, device=DEVICE)
        y = gn(x)
        assert y.shape == (2, 16, 8)

    def test_conv_transpose1d(self):
        ct = nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1).to(DEVICE)
        x = torch.randn(2, 32, 8, device=DEVICE)
        y = ct(x)
        assert y.shape == (2, 16, 16)

    def test_embedding(self):
        emb = nn.Embedding(100, 64).to(DEVICE)
        idx = torch.tensor([0, 5, 99], device=DEVICE)
        y = emb(idx)
        assert y.shape == (3, 64)


class TestFunctionalOps:
    def test_softmax(self):
        x = torch.randn(2, 10, device=DEVICE)
        y = F.softmax(x, dim=-1)
        assert torch.allclose(y.sum(dim=-1), torch.ones(2, device=DEVICE), atol=1e-5)

    def test_grid_sample(self):
        """grid_sample is used in TSDF and feature extraction."""
        input = torch.randn(1, 1, 4, 4, device=DEVICE)
        grid = torch.randn(1, 2, 2, 2, device=DEVICE)
        grid = grid.clamp(-1, 1)
        out = F.grid_sample(input, grid, align_corners=True)
        assert out.shape == (1, 1, 2, 2)

    def test_grid_sample_3d(self):
        """3D grid_sample used for TSDF volume."""
        input = torch.randn(1, 1, 4, 4, 4, device=DEVICE)
        grid = torch.randn(1, 2, 2, 2, 3, device=DEVICE).clamp(-1, 1)
        out = F.grid_sample(input, grid, align_corners=True)
        assert out.shape == (1, 1, 2, 2, 2)

    def test_interpolate(self):
        x = torch.randn(1, 3, 16, 16, device=DEVICE)
        y = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        assert y.shape == (1, 3, 32, 32)

    def test_normalize(self):
        x = torch.randn(2, 3, device=DEVICE)
        y = F.normalize(x, dim=-1)
        norms = y.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2, device=DEVICE), atol=1e-5)

    def test_mse_loss(self):
        pred = torch.randn(2, 3, device=DEVICE)
        target = torch.randn(2, 3, device=DEVICE)
        loss = F.mse_loss(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_log_softmax(self):
        x = torch.randn(4, 10, device=DEVICE)
        y = F.log_softmax(x, dim=-1)
        assert y.shape == (4, 10)


class TestDiffusionOps:
    """Test operations commonly used in the diffusion pipeline."""

    def test_noise_schedule(self):
        from models.helpers import cosine_beta_schedule
        betas = cosine_beta_schedule(100)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        assert (alphas_cumprod >= 0).all()
        assert (alphas_cumprod <= 1).all()

    def test_noise_and_denoise(self):
        """Verify the forward noising process works on device."""
        from models.helpers import cosine_beta_schedule
        betas = cosine_beta_schedule(100).to(DEVICE)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Forward process
        x_0 = torch.randn(2, 3, 32, device=DEVICE)
        t = 50
        noise = torch.randn_like(x_0)
        x_t = sqrt_alphas_cumprod[t] * x_0 + sqrt_one_minus_alphas_cumprod[t] * noise
        assert x_t.shape == x_0.shape
        assert x_t.device.type == DEVICE.type

    def test_gather_on_device(self):
        """Test gather (used for extracting schedule values at timestep t)."""
        schedule = torch.randn(100, device=DEVICE)
        t = torch.tensor([10, 50, 90], device=DEVICE)
        values = schedule.gather(-1, t)
        assert values.shape == (3,)


class TestTensorTransfers:
    """Test CPU <-> Device transfers used throughout the codebase."""

    def test_numpy_to_device(self):
        arr = np.random.randn(3, 3).astype(np.float32)
        t = torch.from_numpy(arr).to(DEVICE)
        assert t.device.type == DEVICE.type

    def test_device_to_numpy(self):
        t = torch.randn(3, 3, device=DEVICE)
        arr = t.cpu().numpy()
        assert isinstance(arr, np.ndarray)

    def test_unsqueeze_and_transfer(self):
        """Common pattern: v.unsqueeze(0).to(device)"""
        t = torch.randn(3, 4)
        t_dev = t.unsqueeze(0).to(DEVICE)
        assert t_dev.shape == (1, 3, 4)
        assert t_dev.device.type == DEVICE.type
