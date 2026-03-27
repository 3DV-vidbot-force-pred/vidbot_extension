"""Tests for dataset utility functions."""
import torch
import numpy as np
import pytest


class TestBackproject:
    def test_backproject_basic(self):
        from diffuser_utils.dataset_utils import backproject
        # Create a simple depth image with known values
        depth = np.ones((4, 4), dtype=np.float32) * 2.0
        intrinsics = np.array([
            [500, 0, 2],
            [0, 500, 2],
            [0, 0, 1],
        ], dtype=np.float32)
        mask = depth > 0
        pts, idxs = backproject(depth, intrinsics, mask, NOCS_convention=False)
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert len(idxs) == 2

    def test_backproject_empty_mask(self):
        from diffuser_utils.dataset_utils import backproject
        depth = np.ones((4, 4), dtype=np.float32) * 2.0
        intrinsics = np.eye(3, dtype=np.float32)
        mask = np.zeros((4, 4), dtype=bool)
        pts, idxs = backproject(depth, intrinsics, mask, NOCS_convention=False)
        assert pts.shape[0] == 0


class TestCropAndPad:
    def test_crop_and_pad_image(self):
        from diffuser_utils.dataset_utils import crop_and_pad_image
        img = np.random.rand(100, 100, 3).astype(np.float32)
        center = np.array([50, 50])
        result = crop_and_pad_image(img, center, scale=40, res=64, channel=3)
        assert result.shape[0] == 64
        assert result.shape[1] == 64


class TestLoadSaveJson:
    def test_load_save_json(self, tmp_path):
        from diffuser_utils.dataset_utils import load_json, save_json
        data = {"key": "value", "num": 42}
        fpath = str(tmp_path / "test.json")
        save_json(fpath, data)
        loaded = load_json(fpath)
        assert loaded == data
