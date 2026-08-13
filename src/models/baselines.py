"""
Baseline models for comparison with TTN architectures.

Includes:
1. Logistic Regression — linear baseline
2. MLP — 2-layer perceptron
3. Lightweight CNN — small convolutional network
4. MPS Classifier — Matrix Product State (1D tensor network baseline)

For fair comparison, MLP and CNN baselines are configured to have
comparable parameter counts to the TTN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LogisticRegressionModel(nn.Module):
    """Simple logistic regression baseline."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron baseline.

    2-layer MLP with configurable hidden dimension.
    For fair comparison, set hidden_dim to match TTN parameter count.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightCNN(nn.Module):
    """
    Lightweight CNN baseline for image classification.

    Architecture: 2 conv layers + 2 FC layers.
    Designed for 28×28 (MNIST/Fashion) or 32×32 (CIFAR) inputs.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calculate FC input size
        fc_size = input_size // 4  # two max pools
        self.fc_input = 32 * fc_size * fc_size

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape from flat to image
        B = x.shape[0]
        x = x.view(B, self.input_channels, self.input_size, self.input_size)
        x = self.features(x)
        x = x.view(B, -1)
        return self.classifier(x)


class MPSClassifier(nn.Module):
    """
    Matrix Product State (MPS) classifier — 1D tensor network baseline.

    An MPS is a linear chain of tensors, contrasted with the TTN's tree
    structure. This serves as the primary tensor network baseline to
    demonstrate the advantage of hierarchical (tree) structure.

    Architecture:
        Input → Feature Map → Chain Contraction → Output

    The MPS contracts features left-to-right along a 1D chain:
        T₁ — T₂ — T₃ — ... — Tₙ

    Each tensor Tᵢ has shape (χ_left, d_i, χ_right) where d_i is the
    local feature dimension and χ is the bond dimension.
    """

    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        bond_dim: int = 8,
        local_dim: int = 2,
        feature_map_config: Optional[dict] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bond_dim = bond_dim
        self.local_dim = local_dim

        # Feature map
        from src.data.feature_maps import get_feature_map
        if feature_map_config is None:
            feature_map_config = {"type": "trigonometric", "local_dim": 2}
        self.feature_map = get_feature_map(feature_map_config)
        self.local_dim = feature_map_config.get("local_dim", 2)

        # MPS tensors
        # First tensor: (d, χ)
        # Middle tensors: (χ, d, χ)
        # Last tensor: (χ, d, num_classes)
        self.tensors = nn.ParameterList()

        # First
        self.tensors.append(nn.Parameter(torch.randn(self.local_dim, bond_dim) * 0.01))

        # Middle
        for i in range(1, input_dim - 1):
            self.tensors.append(
                nn.Parameter(torch.randn(bond_dim, self.local_dim, bond_dim) * 0.01)
            )

        # Last
        self.tensors.append(
            nn.Parameter(torch.randn(bond_dim, self.local_dim, num_classes) * 0.01)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Contract the MPS chain left-to-right.

        Args:
            x: (batch, input_dim)

        Returns:
            (batch, num_classes) logits
        """
        features = self.feature_map(x)  # (B, N, d)

        # First contraction: (B, d) @ (d, χ) → (B, χ)
        state = torch.einsum("bd,dq->bq", features[:, 0, :], self.tensors[0])

        # Middle contractions: (B, χ) × (B, d) × (χ, d, χ) → (B, χ)
        for i in range(1, self.input_dim - 1):
            feat_i = features[:, i, :]  # (B, d)
            tensor_i = self.tensors[i]  # (χ, d, χ)
            state = torch.einsum("bq,bd,qdp->bp", state, feat_i, tensor_i)

        # Last contraction: (B, χ) × (B, d) × (χ, d, C) → (B, C)
        feat_last = features[:, -1, :]
        logits = torch.einsum("bq,bd,qdc->bc", state, feat_last, self.tensors[-1])

        return logits

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_classes={self.num_classes}, "
            f"bond_dim={self.bond_dim}, local_dim={self.local_dim}"
        )
