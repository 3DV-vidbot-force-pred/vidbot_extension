"""Tests for neural network model components."""
import torch
import numpy as np
import pytest
from vidbot_utils.device import get_device

DEVICE = get_device()


class TestHelpers:
    def test_cosine_beta_schedule(self):
        from models.helpers import cosine_beta_schedule
        betas = cosine_beta_schedule(100)
        assert betas.shape == (100,)
        assert (betas >= 0).all()
        assert (betas <= 0.999).all()

    def test_focal_loss(self):
        from models.helpers import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        pred = torch.randn(4, 3)
        target = torch.tensor([0, 1, 2, 1])
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_weighted_l1(self):
        from models.helpers import WeightedL1
        weights = torch.ones(10, 3)
        loss_fn = WeightedL1(weights)
        pred = torch.randn(2, 10, 3)
        targ = torch.randn(2, 10, 3)
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_weighted_l2(self):
        from models.helpers import WeightedL2
        weights = torch.ones(10, 3)
        loss_fn = WeightedL2(weights)
        pred = torch.randn(2, 10, 3)
        targ = torch.randn(2, 10, 3)
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_fourier_positional_encoding(self):
        from models.helpers import fourier_positional_encoding
        x = torch.randn(2, 4, 3, device=DEVICE)
        enc = fourier_positional_encoding(x, L=4)
        assert enc.shape == (2, 4, 3 * 2 * 4)

    def test_ema(self):
        from models.helpers import EMA
        ema = EMA(beta=0.999)
        model_a = torch.nn.Linear(3, 3)
        model_b = torch.nn.Linear(3, 3)
        # Should not raise
        ema.update_model_average(model_a, model_b)

    def test_extract(self):
        from models.helpers import extract
        a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        t = torch.tensor([0, 2, 4])
        result = extract(a, t, (3, 4, 4))
        assert result.shape == (3, 1, 1)


class TestTSDFVolume:
    def test_tsdf_volume_cpu(self):
        from models.helpers import TSDFVolume
        vol_bounds = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
        tsdf = TSDFVolume(vol_bounds, voxel_dim=16, use_gpu=False)
        vol = tsdf.get_tsdf_volume()
        assert vol.shape == (16, 16, 16)
        assert np.allclose(vol, 1.0)  # initialized to ones

    def test_tsdf_volume_gpu(self):
        from models.helpers import TSDFVolume
        vol_bounds = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
        tsdf = TSDFVolume(vol_bounds, voxel_dim=16, use_gpu=True)
        vol = tsdf.get_tsdf_volume()
        assert vol.shape == (16, 16, 16)

    def test_tsdf_volume_vox2world(self):
        from models.helpers import TSDFVolume
        origin = torch.zeros(3)
        coords = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32)
        world = TSDFVolume.vox2world(origin, coords, 0.1)
        assert world.shape == (2, 3)
        assert torch.allclose(world[0], torch.zeros(3))


class TestLayers3D:
    def test_sinusoidal_pos_emb(self):
        from models.layers_3d import SinusoidalPosEmb
        emb = SinusoidalPosEmb(dim=64)
        t = torch.arange(10).float()
        result = emb(t)
        assert result.shape == (10, 64)

    def test_rotary_position_encoding(self):
        from models.layers_3d import RotaryPositionEncoding
        rpe = RotaryPositionEncoding(feature_dim=32)
        pos = torch.randn(2, 16)  # [B, N]
        position_code = rpe(pos)
        # Returns [B, N, feature_dim, 2] (cos and sin stacked)
        assert position_code.shape[0] == 2
        assert position_code.shape[1] == 16
        assert position_code.shape[-1] == 2


class TestScatterMean:
    def test_scatter_mean_replacement(self):
        from models.layers_3d import scatter_mean
        # Simple test: scatter 4 source values into 2 target bins
        src = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [1, 4]
        index = torch.tensor([[0, 0, 1, 1]])  # two bins
        out = torch.zeros(1, 2)
        result = scatter_mean(src, index, out=out, dim=-1)
        # Bin 0: mean(1, 2) = 1.5, Bin 1: mean(3, 4) = 3.5
        assert result.shape == (1, 2)
        assert abs(result[0, 0].item() - 1.5) < 0.01
        assert abs(result[0, 1].item() - 3.5) < 0.01


class TestAttention:
    def test_multi_head_attention(self):
        from models.attention import MultiHeadAttention
        mha = MultiHeadAttention(
            num_heads=4,
            num_q_input_channels=64,
            num_kv_input_channels=64,
        ).to(DEVICE)
        x_q = torch.randn(2, 8, 64, device=DEVICE)
        x_kv = torch.randn(2, 12, 64, device=DEVICE)
        result = mha(x_q, x_kv)
        assert result.last_hidden_state.shape == (2, 8, 64)

    def test_cross_attention_layer(self):
        from models.attention import CrossAttentionLayer
        cal = CrossAttentionLayer(
            num_heads=4,
            num_q_input_channels=64,
            num_kv_input_channels=64,
        ).to(DEVICE)
        x_q = torch.randn(2, 8, 64, device=DEVICE)
        x_kv = torch.randn(2, 12, 64, device=DEVICE)
        result = cal(x_q, x_kv)
        assert result.last_hidden_state.shape == (2, 8, 64)


class TestPerceiver:
    def test_feature_perceiver_instantiation(self):
        from models.perceiver import FeaturePerceiver
        perceiver = FeaturePerceiver(
            transition_dim=3,
            condition_dim=64,
            time_emb_dim=32,
            encoder_q_input_channels=64,
            encoder_kv_input_channels=64,
            encoder_num_heads=4,
        )
        assert perceiver is not None


class TestTemporalUnet:
    def test_forward(self):
        from models.temporal import TemporalMapUnet
        model = TemporalMapUnet(
            horizon=32,
            transition_dim=3,
            cond_dim=64,
            output_dim=3,
            dim=32,
            dim_mults=(1, 2),
        ).to(DEVICE)
        # Input: [batch, horizon, transition_dim]
        x = torch.randn(2, 32, 3, device=DEVICE)
        cond = torch.randn(2, 64, device=DEVICE)
        time = torch.randint(0, 100, (2,), device=DEVICE).float()
        out = model(x, cond, time)
        assert out.shape[0] == 2  # batch
        assert out.shape == (2, 32, 3)  # [batch, horizon, output_dim]


class TestContactPredictor:
    def test_instantiation(self):
        from models.contact import ContactPredictor
        model = ContactPredictor(
            clip_dim=512,
            num_pred_channels=9,
            num_contact_states=4,
        )
        assert model is not None


class TestGoalPredictor:
    def test_instantiation(self):
        from models.goal import GoalPredictor
        model = GoalPredictor(
            clip_dim=512,
            num_pred_channels=2,
        )
        assert model is not None


class TestRotation6D:
    def test_rotation_6d_roundtrip(self):
        from algos.traj_optimizer import rotation_6d_to_matrix, matrix_to_rotation_6d
        # Create a rotation matrix (identity)
        mat = torch.eye(3).unsqueeze(0)  # [1, 3, 3]
        r6d = matrix_to_rotation_6d(mat)
        assert r6d.shape == (1, 6)
        mat_recovered = rotation_6d_to_matrix(r6d)
        assert mat_recovered.shape == (1, 3, 3)
        assert torch.allclose(mat, mat_recovered, atol=1e-5)

    def test_rotation_6d_batch(self):
        from algos.traj_optimizer import rotation_6d_to_matrix, matrix_to_rotation_6d
        # Random rotation matrices
        from scipy.spatial.transform import Rotation
        rots = Rotation.random(5).as_matrix()
        rots_t = torch.tensor(rots, dtype=torch.float32)
        r6d = matrix_to_rotation_6d(rots_t)
        assert r6d.shape == (5, 6)
        rots_recovered = rotation_6d_to_matrix(r6d)
        assert rots_recovered.shape == (5, 3, 3)
        assert torch.allclose(rots_t, rots_recovered, atol=1e-5)
