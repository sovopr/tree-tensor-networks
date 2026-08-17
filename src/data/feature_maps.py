"""
Quantum-inspired feature maps for Tree Tensor Networks.

Three feature map families:
1. Trigonometric (standard) — maps scalar x to [cos(πx/2), sin(πx/2)]
2. Fourier (novel) — multi-scale learnable frequency embeddings
3. POVM (novel) — trainable positive operator-valued measurement embedding

Each feature map takes a batch of flattened input vectors (B, N) and outputs
a batch of local feature tensors (B, N, d) where d is the local dimension.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TrigonometricFeatureMap(nn.Module):
    """
    Standard trigonometric feature map used in tensor network ML literature.

    Maps each scalar feature x_i ∈ [0, 1] to a 2D local state:
        φ(x_i) = [cos(πx_i/2), sin(πx_i/2)]

    This embeds each feature onto the Bloch sphere, creating a quantum-inspired
    representation where the feature value controls the superposition angle.
    """

    def __init__(self, local_dim: int = 2):
        super().__init__()
        if local_dim != 2:
            raise ValueError(
                f"TrigonometricFeatureMap only supports local_dim=2, got {local_dim}. "
                "Use FourierFeatureMap for higher dimensions."
            )
        self.local_dim = local_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features) with values in [0, 1]
        Returns:
            (batch_size, num_features, 2)
        """
        # Clamp to [0, 1] for numerical safety
        x = x.clamp(0.0, 1.0)
        angle = math.pi * x / 2.0  # (B, N)
        cos_part = torch.cos(angle)  # (B, N)
        sin_part = torch.sin(angle)  # (B, N)
        return torch.stack([cos_part, sin_part], dim=-1)  # (B, N, 2)

    def extra_repr(self) -> str:
        return f"local_dim={self.local_dim}"


class FourierFeatureMap(nn.Module):
    """
    Multi-scale Fourier feature map with optionally learnable frequencies.

    Maps each scalar x_i to a 2K-dimensional vector:
        φ(x_i) = [cos(2πσ₁x_i), sin(2πσ₁x_i), ..., cos(2πσₖx_i), sin(2πσₖx_i)]

    where σ_k are frequency scales, either fixed (log-spaced) or learnable.

    NOVEL CONTRIBUTION: No prior TTN work has combined learnable frequency
    scales with hierarchical tensor contraction. The network jointly optimizes
    the embedding frequencies and the tensor parameters.
    """

    def __init__(
        self,
        local_dim: int = 2,
        num_frequencies: int = 4,
        learnable_frequencies: bool = True,
        min_freq: float = 0.5,
        max_freq: float = 8.0,
    ):
        super().__init__()
        assert local_dim % 2 == 0, "local_dim must be even for Fourier features (cos+sin pairs)"
        self.local_dim = local_dim
        self.num_frequencies = num_frequencies
        self.learnable_frequencies = learnable_frequencies

        # The actual number of frequencies used is local_dim // 2
        # (each frequency contributes a cos and sin component)
        effective_num_freq = local_dim // 2

        # Initialize frequencies as log-spaced between min_freq and max_freq
        log_freqs = torch.linspace(
            math.log(min_freq), math.log(max_freq), effective_num_freq
        )
        init_freqs = torch.exp(log_freqs)  # (effective_num_freq,)

        if learnable_frequencies:
            self.log_frequencies = nn.Parameter(log_freqs)
        else:
            self.register_buffer("log_frequencies", log_freqs)

    @property
    def frequencies(self) -> torch.Tensor:
        return torch.exp(self.log_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features) with values in [0, 1]
        Returns:
            (batch_size, num_features, local_dim)
        """
        x = x.clamp(0.0, 1.0)
        freqs = self.frequencies  # (K,)

        # x: (B, N) -> (B, N, 1), freqs: (K,) -> (1, 1, K)
        angles = 2.0 * math.pi * x.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (B, N, K)

        cos_part = torch.cos(angles)  # (B, N, K)
        sin_part = torch.sin(angles)  # (B, N, K)

        # Interleave cos and sin: [cos1, sin1, cos2, sin2, ...]
        features = torch.stack([cos_part, sin_part], dim=-1)  # (B, N, K, 2)
        features = features.reshape(x.shape[0], x.shape[1], -1)  # (B, N, 2K)

        # Normalize to unit norm for numerical stability in contractions
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)

        return features

    def extra_repr(self) -> str:
        return (
            f"local_dim={self.local_dim}, num_frequencies={self.num_frequencies}, "
            f"learnable={self.learnable_frequencies}"
        )


class POVMFeatureMap(nn.Module):
    """
    Trainable Positive Operator-Valued Measurement (POVM) feature map.

    Inspired by recent Born Machine work (2025): instead of a fixed trigonometric
    embedding, we learn a data-dependent embedding via a small neural network
    that outputs a normalized positive vector (like a POVM element).

    Maps each scalar x_i to a d-dimensional vector via:
        φ(x_i) = softmax(W₂ · ReLU(W₁ · x_i + b₁) + b₂)

    The softmax ensures positivity and normalization, which is critical for
    the probabilistic interpretation of the tensor network.

    NOVEL CONTRIBUTION: Trainable POVM embeddings for TTN classification,
    bridging the gap between fixed quantum embeddings and learned representations.
    """

    def __init__(self, povm_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.local_dim = povm_dim
        self.hidden_dim = hidden_dim

        self.embed_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, povm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features) with values in [0, 1]
        Returns:
            (batch_size, num_features, povm_dim)
        """
        B, N = x.shape
        # Process each feature independently through the embedding network
        x_flat = x.reshape(B * N, 1)  # (B*N, 1)
        embedded = self.embed_net(x_flat)  # (B*N, povm_dim)

        # Softmax for positivity and normalization
        embedded = torch.softmax(embedded, dim=-1)

        return embedded.reshape(B, N, self.local_dim)

    def extra_repr(self) -> str:
        return f"local_dim={self.local_dim}, hidden_dim={self.hidden_dim}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_feature_map(config: dict) -> nn.Module:
    """
    Create a feature map from config dict.

    Args:
        config: dict with keys 'type', 'local_dim', and type-specific params.

    Returns:
        Feature map module.
    """
    fmap_type = config.get("type", "trigonometric")

    if fmap_type == "trigonometric":
        return TrigonometricFeatureMap(local_dim=config.get("local_dim", 2))

    elif fmap_type == "fourier":
        return FourierFeatureMap(
            local_dim=config.get("local_dim", 2),
            num_frequencies=config.get("num_frequencies", 4),
            learnable_frequencies=config.get("learnable_frequencies", True),
        )

    elif fmap_type == "povm":
        return POVMFeatureMap(
            povm_dim=config.get("povm_dim", 4),
            hidden_dim=config.get("hidden_dim", 16),
        )

    else:
        raise ValueError(f"Unknown feature map type: {fmap_type}")
