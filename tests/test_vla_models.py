"""Unit tests for MVP-2 instruction encoder and VLA action head components."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import pytest


class _TokenizationResult(dict):
    """Minimal BatchEncoding-like dict that exposes a ``.to()`` helper."""

    def to(self, device: torch.device | str) -> "_TokenizationResult":
        converted = {key: tensor.to(device) for key, tensor in self.items()}
        return _TokenizationResult(converted)


class _FakeTokenizer:
    """Tokenizer stub that encodes text into deterministic input IDs."""

    def __call__(self, texts: str | list[str], **kwargs) -> _TokenizationResult:
        if isinstance(texts, str):
            texts = [texts]
        seq_len = 4
        batch = len(texts)
        input_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        for idx, text in enumerate(texts):
            value = sum(ord(c) for c in text) % 256
            input_ids[idx] = value
        attention_mask = torch.ones_like(input_ids)
        return _TokenizationResult({"input_ids": input_ids, "attention_mask": attention_mask})


class _FakeModel:
    """Model stub that returns repeatable hidden states derived from input IDs."""

    def __init__(self, hidden_size: int = 384) -> None:
        self.hidden_size = hidden_size
        self.call_count = 0
        self._device = torch.device("cpu")
        self._requires_grad = True

    def to(self, device: torch.device | str) -> "_FakeModel":
        self._device = torch.device(device)
        return self

    def eval(self) -> "_FakeModel":
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        self.call_count += 1
        batch, seq = input_ids.shape
        device = self._device
        hidden = input_ids.to(torch.float32).unsqueeze(-1).to(device) / 256.0
        feature_positions = torch.arange(
            1, self.hidden_size + 1, dtype=torch.float32, device=device
        ).view(1, 1, -1)
        seq_positions = torch.arange(1, seq + 1, dtype=torch.float32, device=device).view(1, seq, 1)
        hidden = torch.sin(hidden * feature_positions + seq_positions)
        return SimpleNamespace(last_hidden_state=hidden)

    def parameters(self) -> tuple:
        return ()

    def requires_grad_(self, requires_grad: bool = True) -> "_FakeModel":
        self._requires_grad = requires_grad
        return self


def _patch_transformers(monkeypatch: pytest.MonkeyPatch, model: _FakeModel) -> None:
    """Patch transformers.AutoTokenizer/AutoModel to use our stubs."""

    fake_tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: fake_tokenizer,
    )
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda *args, **kwargs: model,
    )


def test_instruction_encoder_embeds_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """InstructionEncoder must return fixed-shape embeddings and cache repeated texts."""
    fake_model = _FakeModel()
    _patch_transformers(monkeypatch, fake_model)

    from vla_agent.models import InstructionEncoder

    encoder = InstructionEncoder(device="cpu")
    single = encoder.encode("collect wood")
    assert single.shape == (encoder.embed_dim,)
    assert single.dtype == torch.float32

    repeated = encoder.encode("collect wood")
    assert torch.equal(single, repeated)
    assert fake_model.call_count == 1

    batch = encoder.encode_batch(["collect wood", "place table"])
    assert batch.shape == (2, encoder.embed_dim)
    assert encoder.embed_dim == 384

    similarity = F.cosine_similarity(batch[0:1], batch[1:2])
    assert float(similarity) < 1.0, "Different instructions should produce different embeddings"


def test_crafter_vla_forward_and_shapes() -> None:
    """CrafterVLA must fuse vision and text and emit (B, num_actions) logits."""
    from vla_agent.models import CrafterVLA

    torch.manual_seed(0)
    model = CrafterVLA(pretrained=False)
    model.eval()

    batch = 2
    dummy_image = torch.rand(batch, 3, 64, 64)
    dummy_text = torch.rand(batch, model.text_embed_dim)
    with torch.no_grad():
        logits = model(dummy_image, dummy_text)

    assert logits.shape == (batch, model.num_actions)
    assert logits.dtype == torch.float32
    actions = torch.argmax(logits, dim=1)
    assert actions.shape == (batch,)
    assert actions.min() >= 0 and actions.max() < model.num_actions


def test_crafter_vla_resizes_inputs_before_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forward pass must upscale 64x64 frames to 224x224 before the vision kernel."""
    from torchvision.transforms import functional as TF

    from vla_agent.models import CrafterVLA

    original_resize = TF.resize
    calls: list[tuple[int, tuple[int, int]]] = []

    def spy(image: torch.Tensor, size, *args, **kwargs) -> torch.Tensor:
        assert image.shape[-2:] == (64, 64)
        if isinstance(size, int):
            assert size == 224
        else:
            assert tuple(size) == (224, 224)
        calls.append((image.shape[0], tuple(size) if not isinstance(size, int) else (size, size)))
        return original_resize(image, size, *args, **kwargs)

    monkeypatch.setattr("torchvision.transforms.functional.resize", spy)

    model = CrafterVLA(pretrained=False)
    dummy_image = torch.rand(1, 3, 64, 64)
    dummy_text = torch.rand(1, model.text_embed_dim)
    with torch.no_grad():
        _ = model(dummy_image, dummy_text)

    assert calls, "resize must be invoked before backbone"
    assert calls[0][1] == (224, 224)


def test_crafter_vla_uses_text_embeddings() -> None:
    """Different embeddings for the same frame must produce different logits."""
    from vla_agent.models import CrafterVLA

    torch.manual_seed(0)
    model = CrafterVLA(pretrained=False)
    model.eval()

    frame = torch.rand(1, 3, 64, 64)
    text_embed_zero = torch.zeros(1, model.text_embed_dim)
    text_embed_one = torch.ones(1, model.text_embed_dim)

    with torch.no_grad():
        logits_zero = model(frame, text_embed_zero)
        logits_one = model(frame, text_embed_one)

    assert logits_zero.shape == (1, model.num_actions)
    assert logits_one.shape == (1, model.num_actions)
    assert not torch.allclose(logits_zero, logits_one)


def test_action_head_is_only_trainable_module() -> None:
    """Only the action head parameters should require gradients (<500K vs >28M total)."""
    from vla_agent.models import CrafterVLA

    model = CrafterVLA(pretrained=False)
    action_head_ids = {id(p) for p in model.action_head.parameters()}
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 28_000_000

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable)
    assert trainable_param_count < 500_000
    assert trainable_param_count > 0

    for p in model.parameters():
        if id(p) in action_head_ids:
            assert p.requires_grad
        else:
            assert not p.requires_grad


def test_vision_encoder_gradients_remain_none() -> None:
    """Backward pass must only populate grads for the trainable action head."""
    from torch import nn

    from vla_agent.models import CrafterVLA

    model = CrafterVLA(pretrained=False)
    optimizer = torch.optim.Adam(model.action_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dummy_image = torch.rand(2, 3, 64, 64)
    dummy_text = torch.rand(2, model.text_embed_dim)
    dummy_target = torch.zeros(2, dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    logits = model(dummy_image, dummy_text)
    loss = criterion(logits, dummy_target)
    loss.backward()

    action_head_ids = {id(p) for p in model.action_head.parameters()}
    for p in model.parameters():
        if id(p) in action_head_ids:
            assert p.grad is not None
        else:
            assert p.grad is None


def test_crafter_vla_save_load(tmp_path: Path) -> None:
    """Saving and loading the CrafterVLA state dict must produce identical outputs."""
    from vla_agent.models import CrafterVLA

    model = CrafterVLA(pretrained=False)
    model.eval()

    save_path = tmp_path / "vla.pt"
    torch.save(model.state_dict(), str(save_path))

    loaded = CrafterVLA(pretrained=False)
    loaded.load_state_dict(torch.load(str(save_path), map_location="cpu"))
    loaded.eval()

    sample_image = torch.rand(1, 3, 64, 64)
    sample_text = torch.rand(1, loaded.text_embed_dim)
    with torch.no_grad():
        original = model(sample_image, sample_text)
        restored = loaded(sample_image, sample_text)

    torch.testing.assert_close(
        original, restored, msg="Loaded CrafterVLA must match outputs of the original"
    )
