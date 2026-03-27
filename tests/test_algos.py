"""Tests for algorithm modules (Lightning modules)."""
import torch
import pytest
from vidbot_utils.device import get_device
from easydict import EasyDict as edict

DEVICE = get_device()


class TestContactPredictorModule:
    def test_instantiation(self):
        from algos.contact_algos import ContactPredictorModule
        algo_config = edict({
            "model": {
                "clip_dim": 512,
                "num_pred_channels": 9,
                "num_contact_states": 4,
            }
        })
        module = ContactPredictorModule(algo_config)
        assert module is not None
        assert "policy" in module.nets


class TestGoalPredictorModule:
    def test_instantiation(self):
        from algos.goal_algos import GoalPredictorModule
        algo_config = edict({
            "model": {
                "clip_dim": 512,
                "num_pred_channels": 2,
            }
        })
        module = GoalPredictorModule(algo_config)
        assert module is not None
        assert "policy" in module.nets


class TestTrajectoryDiffusionModule:
    def test_instantiation(self):
        from models.diffuser import DiffuserModel
        # Test DiffuserModel directly with minimal config
        model = DiffuserModel(
            horizon=32,
            output_dim=3,
            base_dim=32,
            dim_mults=[1, 2],
            n_timesteps=50,
            use_map_feat_grid=False,
            use_feature_decoder=False,
        )
        assert model is not None
        assert hasattr(model, "model")  # The inner TemporalMapUnet


class TestTrajectoryOptimizer:
    def test_instantiation(self):
        from algos.traj_optimizer import TrajectoryOptimizer
        optimizer = TrajectoryOptimizer(
            resolution=(64, 64),
            device=DEVICE,
        )
        assert optimizer is not None
        assert optimizer.device == DEVICE

    def test_default_device(self):
        from algos.traj_optimizer import TrajectoryOptimizer
        optimizer = TrajectoryOptimizer(resolution=(64, 64))
        assert optimizer.device.type in ("cuda", "mps", "cpu")
