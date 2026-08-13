"""
Augmented Tree Tensor Network (aTTN) with Disentangler Layers.

Extends the standard TTN by inserting unitary disentangler gates between
tree contraction layers. Inspired by the Multi-scale Entanglement
Renormalization Ansatz (MERA) from quantum physics.

The disentanglers allow the network to redistribute correlations before
contraction, dramatically improving representational power for data
with strong inter-feature correlations (e.g., neighboring pixels in images).

Architecture:
    Input → Feature Map → [Disentangle → Contract] × L → Root → Logits

Reference: "The Augmented Tree Tensor Network Cookbook" (2025)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Any

from src.data.feature_maps import get_feature_map
from src.utils.tensor_ops import qr_init, contract_pair, build_binary_tree_structure


class DisentanglerLayer(nn.Module):
    """
    Layer of unitary disentangler gates.

    Each disentangler acts on a pair of neighboring sites and performs
    a unitary rotation in the joint (d × d) space. Implemented as a
    parameterized unitary via the matrix exponential of an anti-symmetric matrix.
    """

    def __init__(self, num_pairs: int, dim: int):
        """
        Args:
            num_pairs: Number of disentangler gates in this layer
            dim: Local dimension of each site
        """
        super().__init__()
        self.num_pairs = num_pairs
        self.dim = dim
        self.joint_dim = dim * dim

        # Parameterize unitaries via anti-symmetric matrices
        # U = exp(A - A^T) is guaranteed to be unitary
        self.anti_sym_params = nn.ParameterList()
        for _ in range(num_pairs):
            # Upper triangle of anti-symmetric matrix
            num_params = self.joint_dim * (self.joint_dim - 1) // 2
            self.anti_sym_params.append(
                nn.Parameter(torch.randn(num_params) * 0.01)
            )

    def _make_unitary(self, params: torch.Tensor) -> torch.Tensor:
        """Construct a unitary matrix from anti-symmetric parameters."""
        d = self.joint_dim
        # Build anti-symmetric matrix
        A = torch.zeros(d, d, device=params.device, dtype=params.dtype)
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                A[i, j] = params[idx]
                A[j, i] = -params[idx]
                idx += 1

        # Matrix exponential gives unitary
        U = torch.matrix_exp(A)
        return U

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply disentanglers to pairs of neighboring sites.

        Disentanglers are applied to overlapping pairs: (0,1), (2,3), ...
        This is the "even" layer pattern. In a full MERA, you'd alternate
        even and odd patterns, but for TTN augmentation one layer suffices.

        Args:
            inputs: List of tensors, each (batch, dim)

        Returns:
            List of tensors with same length and dimensions, but rotated
        """
        outputs = list(inputs)  # copy
        n = len(inputs)

        for pair_idx in range(self.num_pairs):
            left_idx = 2 * pair_idx
            right_idx = 2 * pair_idx + 1

            if right_idx >= n:
                break

            left = outputs[left_idx]   # (B, d)
            right = outputs[right_idx]  # (B, d)
            batch = left.shape[0]

            # Joint state: outer product → (B, d*d)
            joint = torch.einsum("bi,bj->bij", left, right)  # (B, d, d)
            joint_flat = joint.reshape(batch, self.joint_dim)  # (B, d*d)

            # Apply unitary
            U = self._make_unitary(self.anti_sym_params[pair_idx])  # (d*d, d*d)
            rotated = joint_flat @ U.T  # (B, d*d)

            # Split back into two sites via SVD-like factorization
            rotated_tensor = rotated.reshape(batch, self.dim, self.dim)

            # Project back to local dimensions
            # Use the two marginals of the rotated joint state
            outputs[left_idx] = rotated_tensor.sum(dim=-1)    # (B, d) - trace over right
            outputs[right_idx] = rotated_tensor.sum(dim=-2)   # (B, d) - trace over left

            # Normalize for stability
            outputs[left_idx] = outputs[left_idx] / (outputs[left_idx].norm(dim=-1, keepdim=True) + 1e-8)
            outputs[right_idx] = outputs[right_idx] / (outputs[right_idx].norm(dim=-1, keepdim=True) + 1e-8)

        return outputs


class AugmentedTTNLayer(nn.Module):
    """
    Combined disentangler + contraction layer.
    First applies disentanglers, then contracts pairs.
    """

    def __init__(
        self,
        num_nodes: int,
        d_left: int,
        d_right: int,
        chi_out: int,
        use_disentangler: bool = True,
        init_method: str = "qr",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_disentangler = use_disentangler

        # Disentangler (applied before contraction)
        if use_disentangler:
            self.disentangler = DisentanglerLayer(
                num_pairs=num_nodes,
                dim=d_left,  # assumes d_left == d_right
            )

        # Contraction nodes
        self.nodes = nn.ParameterList()
        for _ in range(num_nodes):
            if init_method == "qr":
                tensor = qr_init((d_left, d_right, chi_out))
            else:
                tensor = torch.randn(d_left, d_right, chi_out) * 0.01
            self.nodes.append(nn.Parameter(tensor))

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply disentangler then contract pairs."""
        # Disentangle
        if self.use_disentangler:
            inputs = self.disentangler(inputs)

        # Contract
        outputs = []
        for i in range(self.num_nodes):
            left_idx = 2 * i
            right_idx = 2 * i + 1
            left = inputs[left_idx]
            right = inputs[right_idx] if right_idx < len(inputs) else inputs[left_idx]
            output = contract_pair(left, right, self.nodes[i])
            outputs.append(output)

        return outputs


class AugmentedTTN(nn.Module):
    """
    Augmented Tree Tensor Network classifier.

    Extends standard TTN with unitary disentangler layers between
    each level of the tree. The disentanglers redistribute correlations
    before contraction, improving representation of entangled features.

    This bridges the gap between simple TTN and full MERA, offering
    better expressivity with manageable computational overhead.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        bond_dim: int = 16,
        feature_map_config: Optional[dict] = None,
        init_method: str = "qr",
        use_disentanglers: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bond_dim = bond_dim
        self.use_disentanglers = use_disentanglers

        # Feature map
        if feature_map_config is None:
            feature_map_config = {"type": "trigonometric", "local_dim": 2}
        self.feature_map = get_feature_map(feature_map_config)
        self.local_dim = feature_map_config.get("local_dim", 2)

        # Build tree structure
        self.tree_layers_structure, self.num_padded = build_binary_tree_structure(input_dim)
        self.num_tree_layers = len(self.tree_layers_structure)

        # Build augmented TTN layers
        self.attn_layers = nn.ModuleList()
        d_in = self.local_dim

        for layer_idx in range(self.num_tree_layers):
            num_nodes = len(self.tree_layers_structure[layer_idx])
            chi_out = bond_dim

            # Use disentanglers on all layers except the last (single node)
            use_dis = use_disentanglers and num_nodes > 1

            layer = AugmentedTTNLayer(
                num_nodes=num_nodes,
                d_left=d_in,
                d_right=d_in,
                chi_out=chi_out,
                use_disentangler=use_dis,
                init_method=init_method,
            )
            self.attn_layers.append(layer)
            d_in = chi_out

        # Classification head
        self.classifier = nn.Linear(bond_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through augmented TTN.

        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, num_classes) logits
        """
        batch_size = x.shape[0]

        # Feature map
        features = self.feature_map(x)
        feature_list = [features[:, i, :] for i in range(features.shape[1])]

        # Pad
        while len(feature_list) < self.num_padded:
            pad = torch.ones(batch_size, self.local_dim, device=x.device) / math.sqrt(self.local_dim)
            feature_list.append(pad)

        # Contract through augmented tree
        current = feature_list
        for layer in self.attn_layers:
            current = layer(current)

        # Classify
        root = current[0]
        logits = self.classifier(root)
        return logits

    def get_intermediate_states(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Get intermediate states for analysis."""
        batch_size = x.shape[0]
        features = self.feature_map(x)
        feature_list = [features[:, i, :] for i in range(features.shape[1])]

        while len(feature_list) < self.num_padded:
            pad = torch.ones(batch_size, self.local_dim, device=x.device) / math.sqrt(self.local_dim)
            feature_list.append(pad)

        all_states = [feature_list]
        current = feature_list
        for layer in self.attn_layers:
            current = layer(current)
            all_states.append(current)

        return all_states

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_classes={self.num_classes}, "
            f"bond_dim={self.bond_dim}, disentanglers={self.use_disentanglers}, "
            f"num_layers={self.num_tree_layers}"
        )
