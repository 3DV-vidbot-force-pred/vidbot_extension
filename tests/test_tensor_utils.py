"""Tests for tensor utility functions."""
import torch
import numpy as np
import pytest
from diffuser_utils.tensor_utils import (
    map_tensor, map_ndarray, clone, detach, to_batch, to_device,
    to_tensor, to_numpy, to_float, to_torch, to_list,
    flatten_single, unsqueeze, squeeze, contiguous,
    pad_sequence_single, slice_tensor_single,
)
from vidbot_utils.device import get_device


DEVICE = get_device()


class TestMapFunctions:
    def test_map_tensor_on_dict(self):
        x = {"a": torch.tensor([1.0, 2.0]), "b": None}
        result = map_tensor(x, lambda t: t * 2)
        assert torch.allclose(result["a"], torch.tensor([2.0, 4.0]))
        assert result["b"] is None

    def test_map_tensor_on_nested_dict(self):
        x = {"a": {"b": torch.tensor([1.0])}}
        result = map_tensor(x, lambda t: t + 1)
        assert torch.allclose(result["a"]["b"], torch.tensor([2.0]))

    def test_map_tensor_on_list(self):
        x = [torch.tensor([1.0]), torch.tensor([2.0])]
        result = map_tensor(x, lambda t: t * 3)
        assert torch.allclose(result[0], torch.tensor([3.0]))
        assert torch.allclose(result[1], torch.tensor([6.0]))

    def test_map_ndarray(self):
        x = {"a": np.array([1.0, 2.0]), "b": None}
        result = map_ndarray(x, lambda a: a * 2)
        np.testing.assert_array_equal(result["a"], np.array([2.0, 4.0]))


class TestCloneDetach:
    def test_clone_tensor(self):
        orig = {"x": torch.tensor([1.0, 2.0])}
        cloned = clone(orig)
        cloned["x"][0] = 99.0
        assert orig["x"][0] == 1.0  # original unchanged

    def test_clone_ndarray(self):
        orig = {"x": np.array([1.0, 2.0])}
        cloned = clone(orig)
        cloned["x"][0] = 99.0
        assert orig["x"][0] == 1.0

    def test_detach(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = detach({"a": t})
        assert not result["a"].requires_grad


class TestDeviceTransfer:
    def test_to_device(self):
        x = {"a": torch.tensor([1.0, 2.0])}
        result = to_device(x, DEVICE)
        assert result["a"].device.type == DEVICE.type

    def test_to_torch(self):
        # Use float32 explicitly since MPS doesn't support float64
        x = {"a": np.array([1.0, 2.0], dtype=np.float32)}
        result = to_torch(x, DEVICE)
        assert isinstance(result["a"], torch.Tensor)
        assert result["a"].device.type == DEVICE.type

    def test_to_torch_float64_fails_on_mps(self):
        """Float64 numpy arrays need float32 conversion for MPS."""
        import platform
        x = {"a": np.array([1.0, 2.0])}  # float64 by default
        if DEVICE.type == "mps":
            with pytest.raises(TypeError):
                to_torch(x, DEVICE)
        else:
            result = to_torch(x, DEVICE)
            assert result["a"].device.type == DEVICE.type


class TestConversions:
    def test_to_tensor(self):
        x = {"a": np.array([1.0, 2.0])}
        result = to_tensor(x)
        assert isinstance(result["a"], torch.Tensor)

    def test_to_numpy(self):
        x = {"a": torch.tensor([1.0, 2.0])}
        result = to_numpy(x)
        assert isinstance(result["a"], np.ndarray)

    def test_to_numpy_from_device(self):
        x = {"a": torch.tensor([1.0, 2.0], device=DEVICE)}
        result = to_numpy(x)
        assert isinstance(result["a"], np.ndarray)
        np.testing.assert_array_almost_equal(result["a"], [1.0, 2.0])

    def test_to_list(self):
        x = {"a": torch.tensor([1.0, 2.0])}
        result = to_list(x)
        assert result["a"] == [1.0, 2.0]

    def test_to_list_from_device(self):
        x = {"a": torch.tensor([1.0, 2.0], device=DEVICE)}
        result = to_list(x)
        assert result["a"] == pytest.approx([1.0, 2.0])

    def test_to_float(self):
        x = {"a": torch.tensor([1, 2], dtype=torch.int32)}
        result = to_float(x)
        assert result["a"].dtype == torch.float32

    def test_to_batch(self):
        x = {"a": torch.randn(3, 4)}
        result = to_batch(x)
        assert result["a"].shape == (1, 3, 4)


class TestReshapeOps:
    def test_flatten_single(self):
        t = torch.randn(2, 3, 4)
        result = flatten_single(t, begin_axis=1)
        assert result.shape == (2, 12)

    def test_unsqueeze(self):
        x = {"a": torch.randn(3, 4)}
        result = unsqueeze(x, 0)
        assert result["a"].shape == (1, 3, 4)

    def test_squeeze(self):
        x = {"a": torch.randn(1, 3, 4)}
        result = squeeze(x, 0)
        assert result["a"].shape == (3, 4)

    def test_contiguous(self):
        t = torch.randn(3, 4).T  # non-contiguous
        result = contiguous({"a": t})
        assert result["a"].is_contiguous()


class TestPadSlice:
    def test_pad_sequence_begin(self):
        seq = torch.randn(5, 3)
        result = pad_sequence_single(seq, padding=(2, 0), pad_values=0.0)
        assert result.shape == (7, 3)

    def test_pad_sequence_end(self):
        seq = torch.randn(5, 3)
        result = pad_sequence_single(seq, padding=(0, 3), pad_values=0.0)
        assert result.shape == (8, 3)

    def test_slice_tensor(self):
        t = torch.arange(10).float()
        result = slice_tensor_single(t, dim=0, start_idx=2, end_idx=5)
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([2.0, 3.0, 4.0]))
