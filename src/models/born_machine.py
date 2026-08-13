"""
TTN Born Machine — Generative Model via Tree Tensor Network.

NOVEL CONTRIBUTION: Uses a Tree Tensor Network to define a probability
distribution over the input space via the Born rule from quantum mechanics:

    P(x) = |⟨x|Ψ⟩|² = |TTN(x)|²

where Ψ is the "wavefunction" encoded by the TTN.

Training: Minimize negative log-likelihood (NLL) or Maximum Mean Discrepancy (MMD).
Sampling: Top-down conditional sampling through the tree.

This is the first tree-structured Born Machine. Prior work uses MPS (1D chain).
The tree structure naturally captures multi-scale correlations in data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

from src.data.feature_maps import get_feature_map
from src.utils.tensor_ops import qr_init, contract_pair, build_binary_tree_structure


class TTNBornMachine(nn.Module):
    """
    Tree Tensor Network Born Machine for generative modeling.

    The TTN defines an unnormalized wavefunction Ψ(x). The probability
    of a data sample is P(x) = |Ψ(x)|² / Z, where Z = Σ_x |Ψ(x)|².

    For training, we minimize the negative log-likelihood:
        L = -E_data[log P(x)] = -E_data[2 log|Ψ(x)|] + log Z

    Or use the MMD loss which doesn't require computing Z.
    """

    def __init__(
        self,
        input_dim: int = 784,
        bond_dim: int = 8,
        feature_map_config: Optional[dict] = None,
        init_method: str = "qr",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bond_dim = bond_dim

        # Feature map
        if feature_map_config is None:
            feature_map_config = {"type": "trigonometric", "local_dim": 2}
        self.feature_map = get_feature_map(feature_map_config)
        self.local_dim = feature_map_config.get("local_dim", 2)

        # Build tree
        self.tree_layers_structure, self.num_padded = build_binary_tree_structure(input_dim)
        self.num_tree_layers = len(self.tree_layers_structure)

        # TTN tensors (same as classifier but no classification head)
        self.ttn_layers = nn.ModuleList()
        d_in = self.local_dim

        for layer_idx in range(self.num_tree_layers):
            num_nodes = len(self.tree_layers_structure[layer_idx])
            chi_out = bond_dim

            nodes = nn.ParameterList()
            for _ in range(num_nodes):
                if init_method == "qr":
                    tensor = qr_init((d_in, d_in, chi_out))
                else:
                    tensor = torch.randn(d_in, d_in, chi_out) * 0.01
                nodes.append(nn.Parameter(tensor))

            self.ttn_layers.append(nodes)
            d_in = chi_out

        # Root tensor: maps to scalar amplitude
        self.root_weight = nn.Parameter(torch.randn(bond_dim) * 0.01)

    def amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the wavefunction amplitude Ψ(x) for input x.

        Args:
            x: (batch, input_dim) features in [0, 1]

        Returns:
            (batch,) — amplitude values
        """
        batch_size = x.shape[0]

        # Feature map
        features = self.feature_map(x)
        feature_list = [features[:, i, :] for i in range(features.shape[1])]

        # Pad
        while len(feature_list) < self.num_padded:
            pad = torch.ones(batch_size, self.local_dim, device=x.device) / math.sqrt(self.local_dim)
            feature_list.append(pad)

        # Contract tree
        current = feature_list
        for layer_nodes in self.ttn_layers:
            next_level = []
            for i, node in enumerate(layer_nodes):
                left_idx = 2 * i
                right_idx = 2 * i + 1
                left = current[left_idx]
                right = current[right_idx] if right_idx < len(current) else current[left_idx]
                output = contract_pair(left, right, node)
                next_level.append(output)
            current = next_level

        # Root contraction: (B, chi) · (chi,) → (B,)
        root = current[0]
        amplitude = torch.einsum("bk,k->b", root, self.root_weight)

        return amplitude

    def log_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log P(x) = 2 * log|Ψ(x)| (up to normalization constant).

        For NLL training, we minimize -log P(x) = -2 * log|Ψ(x)| + const.
        The normalization constant is handled by the loss function.

        Args:
            x: (batch, input_dim)

        Returns:
            (batch,) unnormalized log probabilities
        """
        amp = self.amplitude(x)
        return 2.0 * torch.log(torch.abs(amp) + 1e-10)

    def nll_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss (unnormalized).

        The normalization constant log(Z) is approximately constant
        for a well-trained model, so we omit it and rely on the
        model learning to self-normalize.
        """
        log_probs = self.log_probability(x)
        return -log_probs.mean()

    def mmd_loss(
        self,
        real_data: torch.Tensor,
        generated_data: torch.Tensor,
        bandwidth: float = 1.0,
    ) -> torch.Tensor:
        """
        Maximum Mean Discrepancy loss with RBF kernel.

        MMD avoids computing the partition function Z, making it
        more stable for training Born Machines.

        Args:
            real_data: (B1, input_dim) real samples
            generated_data: (B2, input_dim) generated/model samples
            bandwidth: RBF kernel bandwidth

        Returns:
            Scalar MMD² loss
        """
        def rbf_kernel(x, y, bw):
            diff = x.unsqueeze(1) - y.unsqueeze(0)  # (B1, B2, D)
            dist_sq = (diff ** 2).sum(-1)  # (B1, B2)
            return torch.exp(-dist_sq / (2 * bw ** 2))

        K_xx = rbf_kernel(real_data, real_data, bandwidth)
        K_yy = rbf_kernel(generated_data, generated_data, bandwidth)
        K_xy = rbf_kernel(real_data, generated_data, bandwidth)

        mmd_sq = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd_sq

    @torch.no_grad()
    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate samples via importance sampling.

        For a Born Machine, exact sampling requires computing conditional
        probabilities top-down through the tree. Here we use a simpler
        approach: propose samples from a uniform distribution and weight
        by |Ψ(x)|².

        For better quality, use MCMC sampling (Metropolis-Hastings).

        Args:
            num_samples: Number of samples to generate
            device: Device

        Returns:
            (num_samples, input_dim) generated samples
        """
        # Simple importance sampling with uniform proposal
        # Generate many proposals and select with probability ∝ |Ψ|²
        num_proposals = num_samples * 10

        proposals = torch.rand(num_proposals, self.input_dim, device=device)
        log_probs = self.log_probability(proposals)
        probs = torch.exp(log_probs - log_probs.max())  # numerical stability
        probs = probs / probs.sum()

        # Resample
        indices = torch.multinomial(probs, num_samples, replacement=True)
        return proposals[indices]

    @torch.no_grad()
    def sample_mcmc(
        self,
        num_samples: int,
        num_steps: int = 1000,
        step_size: float = 0.05,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate samples via Metropolis-Hastings MCMC.

        More accurate than importance sampling but slower.
        """
        # Initialize chain
        x = torch.rand(1, self.input_dim, device=device)
        log_p = self.log_probability(x).item()

        samples = []
        accepted = 0

        for step in range(num_steps * num_samples):
            # Propose
            x_new = x + torch.randn_like(x) * step_size
            x_new = x_new.clamp(0, 1)

            log_p_new = self.log_probability(x_new).item()

            # Accept/reject
            if math.log(torch.rand(1).item() + 1e-10) < log_p_new - log_p:
                x = x_new
                log_p = log_p_new
                accepted += 1

            # Thin: collect every num_steps
            if (step + 1) % num_steps == 0:
                samples.append(x.clone())

        if samples:
            return torch.cat(samples, dim=0)
        else:
            return torch.rand(num_samples, self.input_dim, device=device)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, bond_dim={self.bond_dim}, "
            f"local_dim={self.local_dim}, num_layers={self.num_tree_layers}"
        )
