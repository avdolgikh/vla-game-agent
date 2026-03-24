"""Model architectures for Crafter agents (MVP-1 CNN, MVP-2 VLA)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from transformers import AutoModel, AutoTokenizer


class CrafterCNN(nn.Module):
    """Minimal CNN encoder for 64x64 RGB observations -> action logits."""

    def __init__(self, num_actions: int = 8) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_actions)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class InstructionEncoder:
    """Encode instruction strings via frozen all-MiniLM-L6-v2 embeddings."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        config = getattr(self.model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        self._embed_dim = int(hidden_size) if hidden_size is not None else 384
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, text: str) -> torch.Tensor:
        batch = self.encode_batch([text])
        return batch[0]

    def encode_batch(self, texts: Sequence[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty((0, self.embed_dim), dtype=torch.float32, device=self.device)

        results: list[torch.Tensor | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        for idx, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is None:
                missing_texts.append(text)
                missing_indices.append(idx)
            else:
                results[idx] = cached

        if missing_texts:
            embeddings = self._encode_texts(missing_texts)
            for offset, idx in enumerate(missing_indices):
                embedding = embeddings[offset].detach().clone()
                self._cache[texts[idx]] = embedding
                results[idx] = embedding

        final: list[torch.Tensor] = []
        for tensor in results:
            if tensor is None:
                raise RuntimeError("Instruction embedding missing from cache.")
            final.append(tensor)
        return torch.stack(final)

    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokenized = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**tokenized)
        embeddings = self._mean_pool(output.last_hidden_state, tokenized["attention_mask"])
        return embeddings.to(torch.float32)

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / counts


class CrafterVLA(nn.Module):
    """Fuse frozen ConvNeXt vision embeddings with precomputed text embeddings."""

    def __init__(
        self,
        text_embed_dim: int = 384,
        num_actions: int = 8,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.text_embed_dim = text_embed_dim
        self.num_actions = num_actions
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.vision_backbone = backbone.features
        self.vision_avgpool = backbone.avgpool
        self.vision_norm = backbone.classifier[0]
        self.vision_dim = 768
        self._freeze_module(self.vision_backbone)
        self._freeze_module(self.vision_norm)
        self.action_head = nn.Sequential(
            nn.Linear(self.vision_dim + text_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "CrafterVLA":
        """Override to keep frozen vision modules in eval mode.

        ConvNeXt contains StochasticDepth layers that randomly drop residual
        connections in training mode.  The frozen backbone must stay in eval
        mode so features are deterministic and noise-free.
        """
        super().train(mode)
        # Always keep frozen vision components in eval mode.
        self.vision_backbone.eval()
        self.vision_norm.eval()
        return self

    def forward(self, image: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError("image tensor must be 4D (B, C, H, W)")
        if text_embed.dim() != 2:
            raise ValueError("text_embed tensor must be 2D (B, D)")
        if text_embed.shape[1] != self.text_embed_dim:
            raise ValueError(
                f"text_embed dimension mismatch: expected {self.text_embed_dim}, "
                f"got {text_embed.shape[1]}"
            )

        image = image.to(self.image_mean.dtype)
        image = (image - self.image_mean) / self.image_std
        text_embed = text_embed.to(image.device, dtype=torch.float32)

        with torch.no_grad():
            vision = self.vision_backbone(image)
            vision = self.vision_avgpool(vision)
            vision = self.vision_norm(vision)
        vision = torch.flatten(vision, 1)
        if vision.shape[1] != self.vision_dim:
            raise RuntimeError(
                f"Unexpected vision feature dimension {vision.shape[1]}, expected {self.vision_dim}."
            )
        fused = torch.cat([vision, text_embed], dim=1)
        logits = self.action_head(fused)
        return logits
