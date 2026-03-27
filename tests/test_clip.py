"""Tests for CLIP model loading and inference."""
import torch
import pytest
from vidbot_utils.device import get_device

DEVICE = get_device()


class TestCLIPTokenizer:
    def test_tokenize(self):
        from models.clip import tokenize
        tokens = tokenize("open the drawer")
        assert isinstance(tokens, torch.Tensor)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 1

    def test_tokenize_batch(self):
        from models.clip import tokenize
        tokens = tokenize(["open drawer", "pick up mug"])
        assert tokens.shape[0] == 2

    def test_tokenize_empty(self):
        from models.clip import tokenize
        tokens = tokenize("")
        assert tokens.shape[0] == 1


class TestCLIPModel:
    @pytest.fixture(scope="class")
    def clip_model(self):
        from models.clip import clip
        model, transform = clip.load("ViT-B/16", device=DEVICE, jit=False)
        model.float()
        model.eval()
        return model, transform

    def test_load_model(self, clip_model):
        model, transform = clip_model
        assert model is not None
        assert transform is not None

    def test_text_encoding(self, clip_model):
        from models.clip import tokenize
        model, _ = clip_model
        tokens = tokenize("open the drawer").to(DEVICE)
        with torch.no_grad():
            features = model.encode_text(tokens)
        assert features.shape[0] == 1
        assert features.shape[1] == 512  # ViT-B output dim

    def test_text_encoding_batch(self, clip_model):
        from models.clip import tokenize
        model, _ = clip_model
        tokens = tokenize(["open drawer", "pick up mug", "close lid"]).to(DEVICE)
        with torch.no_grad():
            features = model.encode_text(tokens)
        assert features.shape[0] == 3
        assert features.shape[1] == 512

    def test_null_text_embeddings(self, clip_model):
        from models.helpers import compute_null_text_embeddings
        model, _ = clip_model
        null_emb = compute_null_text_embeddings(model, batch_size=2, device=DEVICE)
        assert null_emb.shape == (2, 512)
