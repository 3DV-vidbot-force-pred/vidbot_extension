"""Tests for diffusion guidance and parameters."""
import torch
import pytest
from vidbot_utils.device import get_device

DEVICE = get_device()


class TestGuidanceParams:
    def test_params_dict_exists(self):
        from diffuser_utils.guidance_params import GUIDANCE_PARAMS_DICT
        assert isinstance(GUIDANCE_PARAMS_DICT, dict)
        assert len(GUIDANCE_PARAMS_DICT) > 0

    def test_params_have_required_keys(self):
        from diffuser_utils.guidance_params import GUIDANCE_PARAMS_DICT
        required_keys = [
            "goal_weight", "noncollide_weight", "normal_weight",
            "contact_weight", "fine_voxel_resolution", "exclude_object_points"
        ]
        for action, params in GUIDANCE_PARAMS_DICT.items():
            for key in required_keys:
                assert key in params, f"Missing key {key} in params for action {action}"

    def test_common_actions(self):
        from diffuser_utils.guidance_params import COMMON_ACTIONS
        assert isinstance(COMMON_ACTIONS, (list, tuple, set))
        assert len(COMMON_ACTIONS) > 0


class TestDiffuserGuidance:
    def test_instantiation(self):
        from diffuser_utils.guidance_loss import DiffuserGuidance
        guidance = DiffuserGuidance(
            goal_weight=1.0,
            noncollide_weight=1.0,
            contact_weight=1.0,
            normal_weight=1.0,
            scale=1.0,
        )
        assert guidance is not None

    def test_default_params(self):
        from diffuser_utils.guidance_loss import DiffuserGuidance
        guidance = DiffuserGuidance()
        assert guidance is not None
