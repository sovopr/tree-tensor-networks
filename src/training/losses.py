"""
Custom loss functions for TTN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TTNLoss(nn.Module):
    """
    Combined loss for TTN classifier with optional regularization.

    L = CrossEntropy + λ_orth * Orthogonality_Regularizer

    The orthogonality regularizer encourages isometric tensors,
    which improves training stability and has physical motivation
    (isometric TNs correspond to proper quantum channels).
    """

    def __init__(
        self,
        lambda_orth: float = 0.01,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_orth = lambda_orth
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def orthogonality_regularizer(self, model: nn.Module) -> torch.Tensor:
        """
        Penalize deviation from isometric tensors.

        For a TTN node tensor T of shape (d_l, d_r, χ), the isometry
        condition is: Σ_{ij} T_{ij,k} T_{ij,k'} = δ_{kk'}

        We compute ||T^T T - I||² as the regularization loss.
        """
        reg = torch.tensor(0.0, device=next(model.parameters()).device)

        for param in model.parameters():
            if param.dim() == 3:  # TTN node tensor (d_l, d_r, chi)
                d_l, d_r, chi = param.shape
                # Reshape to (d_l * d_r, chi)
                mat = param.reshape(d_l * d_r, chi)
                # T^T T should be identity
                gram = mat.T @ mat
                identity = torch.eye(chi, device=param.device)
                reg = reg + ((gram - identity) ** 2).sum()

        return reg

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute total loss.

        Args:
            logits: (batch, num_classes)
            targets: (batch,) integer labels
            model: Optional model for regularization

        Returns:
            Scalar loss
        """
        loss = self.ce_loss(logits, targets)

        if model is not None and self.lambda_orth > 0:
            loss = loss + self.lambda_orth * self.orthogonality_regularizer(model)

        return loss


class BornMachineLoss(nn.Module):
    """
    Loss for training Born Machine generative models.

    Supports:
    1. Negative log-likelihood (NLL)
    2. Maximum Mean Discrepancy (MMD) with RBF kernel
    3. Combined NLL + MMD
    """

    def __init__(
        self,
        loss_type: str = "nll",
        mmd_bandwidth: float = 1.0,
        nll_weight: float = 1.0,
        mmd_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.mmd_bandwidth = mmd_bandwidth
        self.nll_weight = nll_weight
        self.mmd_weight = mmd_weight

    def forward(self, model, real_data: torch.Tensor) -> torch.Tensor:
        """
        Compute Born Machine loss.

        Args:
            model: TTNBornMachine instance
            real_data: (batch, input_dim) real data samples

        Returns:
            Scalar loss
        """
        if self.loss_type == "nll":
            return self.nll_weight * model.nll_loss(real_data)

        elif self.loss_type == "mmd":
            # Generate samples from the model
            with torch.no_grad():
                generated = model.sample(real_data.shape[0], device=real_data.device)
            return self.mmd_weight * model.mmd_loss(real_data, generated, self.mmd_bandwidth)

        elif self.loss_type == "combined":
            nll = model.nll_loss(real_data)
            with torch.no_grad():
                generated = model.sample(real_data.shape[0], device=real_data.device)
            mmd = model.mmd_loss(real_data, generated, self.mmd_bandwidth)
            return self.nll_weight * nll + self.mmd_weight * mmd

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
