"""
Tests for ml/deepfake/model.py's PyTorch CNN baseline: forward-pass
shape correctness and checkpoint save/load round-tripping.

Requires torch. In environments without torch installed, these tests
are skipped (not failed) via pytest.importorskip -- the ImportError
fallback behavior itself is covered separately in
test_model_without_torch.py, which runs everywhere.
"""
import pytest

torch = pytest.importorskip("torch")

from ml.deepfake.model import SyntheticVoiceCNN, save_checkpoint, load_checkpoint


def test_cnn_forward_pass_output_shape():
    model = SyntheticVoiceCNN(n_mels=64)
    batch = torch.randn(4, 1, 64, 128)
    logits = model(batch)
    assert logits.shape == (4,)


def test_cnn_forward_pass_single_sample():
    model = SyntheticVoiceCNN(n_mels=64)
    single = torch.randn(1, 1, 64, 128)
    logits = model(single)
    assert logits.shape == (1,)


def test_cnn_output_is_finite():
    model = SyntheticVoiceCNN(n_mels=64)
    batch = torch.randn(2, 1, 64, 128)
    logits = model(batch)
    assert torch.isfinite(logits).all()


def test_cnn_sigmoid_output_in_valid_range():
    model = SyntheticVoiceCNN(n_mels=64)
    model.eval()
    batch = torch.randn(3, 1, 64, 128)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch))
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    model = SyntheticVoiceCNN(n_mels=64)
    config = {"n_mels": 64, "fixed_frames": 128, "model_version": "cnn-baseline-v1-test"}
    checkpoint_path = tmp_path / "test_checkpoint.pt"

    save_checkpoint(model, checkpoint_path, config)
    assert checkpoint_path.exists()

    loaded_model, loaded_config = load_checkpoint(checkpoint_path)
    assert loaded_config["model_version"] == "cnn-baseline-v1-test"
    assert loaded_config["n_mels"] == 64

    # Same input should produce the same output before and after save/load.
    model.eval()
    loaded_model.eval()
    sample = torch.randn(1, 1, 64, 128)
    with torch.no_grad():
        original_output = model(sample)
        loaded_output = loaded_model(sample)
    assert torch.allclose(original_output, loaded_output, atol=1e-6)


def test_model_has_reasonable_parameter_count():
    """Sanity check that this is genuinely a 'small baseline' model, not
    an accidentally huge network that would be slow to train/ship."""
    model = SyntheticVoiceCNN(n_mels=64)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 500_000, f"Expected a small baseline model, got {n_params} parameters"
