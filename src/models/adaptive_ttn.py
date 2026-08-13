"""
Adaptive Tree Tensor Network with Learnable Topology.

NOVEL CONTRIBUTION: Instead of a fixed binary tree, this model learns
the optimal tree structure from data using two complementary strategies:

1. Mutual Information (MI) Guided Initialization:
   Before training, compute pairwise MI between features.
   Use hierarchical clustering to determine initial pairing order.

2. Differentiable Architecture Search via Gumbel-Softmax:
   During training, soft-assign features to tree nodes using
   Gumbel-Softmax relaxation. The topology is optimized jointly
   with the tensor parameters.

This is the first data-driven tree structure for tensor network
classification. Prior work uses fixed binary trees with arbitrary
feature ordering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any, Tuple

from src.data.feature_maps import get_feature_map
from src.utils.tensor_ops import (
    qr_init,
    contract_pair,
    build_binary_tree_structure,
    compute_mutual_information,
    mi_guided_pairing,
)


class GumbelSoftmaxPairing(nn.Module):
    """
    Differentiable feature pairing via Gumbel-Softmax.

    For N features at a given level, learns a soft assignment matrix
    that determines which features are paired together. During training,
    uses Gumbel-Softmax for differentiable discrete selection.
    At inference, uses argmax (hard assignment).

    The assignment matrix A ∈ R^(N/2 × N) where A[i,:] is a softmax
    distribution over features, selecting which two features form pair i.
    """

    def __init__(self, num_features: int, temperature: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.num_pairs = num_features // 2
        self.temperature = temperature

        # Learnable logits for feature-to-pair assignment
        # Shape: (num_pairs, 2, num_features)
        # For each pair, we have two slots (left, right), each selects a feature
        self.assignment_logits = nn.Parameter(
            torch.randn(self.num_pairs, 2, num_features) * 0.1
        )

    def forward(
        self,
        features: List[torch.Tensor],
        hard: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute soft pairing of features.

        Args:
            features: List of N tensors, each (batch, d)
            hard: If True, use hard argmax instead of soft Gumbel-Softmax

        Returns:
            (left_features, right_features): Each a list of N/2 tensors
        """
        N = len(features)
        batch_size = features[0].shape[0]
        d = features[0].shape[1]

        # Stack features: (B, N, d)
        feat_stack = torch.stack(features, dim=1)

        left_list = []
        right_list = []

        for pair_idx in range(self.num_pairs):
            for slot in range(2):
                logits = self.assignment_logits[pair_idx, slot]  # (N,)

                if hard or not self.training:
                    # Hard selection
                    idx = logits.argmax()
                    selected = feat_stack[:, idx, :]  # (B, d)
                else:
                    # Gumbel-Softmax
                    weights = F.gumbel_softmax(
                        logits.unsqueeze(0).expand(batch_size, -1),
                        tau=self.temperature,
                        hard=False,
                        dim=-1,
                    )  # (B, N)

                    # Weighted sum of features
                    selected = torch.bmm(
                        weights.unsqueeze(1),  # (B, 1, N)
                        feat_stack,            # (B, N, d)
                    ).squeeze(1)  # (B, d)

                if slot == 0:
                    left_list.append(selected)
                else:
                    right_list.append(selected)

        return left_list, right_list

    def get_hard_assignments(self) -> List[Tuple[int, int]]:
        """Get the current hard feature pairing."""
        pairs = []
        for pair_idx in range(self.num_pairs):
            left_idx = self.assignment_logits[pair_idx, 0].argmax().item()
            right_idx = self.assignment_logits[pair_idx, 1].argmax().item()
            pairs.append((left_idx, right_idx))
        return pairs


class AdaptiveTTNLayer(nn.Module):
    """
    Adaptive TTN layer with optional Gumbel-Softmax pairing.
    """

    def __init__(
        self,
        num_nodes: int,
        num_inputs: int,
        d_left: int,
        d_right: int,
        chi_out: int,
        adaptive: bool = True,
        temperature: float = 1.0,
        init_method: str = "qr",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.adaptive = adaptive

        # Adaptive pairing
        if adaptive and num_inputs > 2:
            self.pairing = GumbelSoftmaxPairing(num_inputs, temperature)
        else:
            self.pairing = None

        # Contraction nodes
        self.nodes = nn.ParameterList()
        for _ in range(num_nodes):
            if init_method == "qr":
                tensor = qr_init((d_left, d_right, chi_out))
            else:
                tensor = torch.randn(d_left, d_right, chi_out) * 0.01
            self.nodes.append(nn.Parameter(tensor))

    def forward(
        self,
        inputs: List[torch.Tensor],
        hard: bool = False,
    ) -> List[torch.Tensor]:
        """Contract with adaptive or fixed pairing."""
        if self.pairing is not None:
            left_list, right_list = self.pairing(inputs, hard=hard)
            outputs = []
            for i in range(self.num_nodes):
                output = contract_pair(left_list[i], right_list[i], self.nodes[i])
                outputs.append(output)
        else:
            # Fixed sequential pairing
            outputs = []
            for i in range(self.num_nodes):
                left_idx = 2 * i
                right_idx = 2 * i + 1
                left = inputs[left_idx]
                right = inputs[right_idx] if right_idx < len(inputs) else inputs[left_idx]
                output = contract_pair(left, right, self.nodes[i])
                outputs.append(output)

        return outputs


class AdaptiveTTN(nn.Module):
    """
    Adaptive Tree Tensor Network with learnable topology.

    Combines:
    1. MI-guided initialization of tree structure
    2. Gumbel-Softmax differentiable pairing during training
    3. Temperature annealing for convergence to hard assignments

    This is the most novel component of the project — the first
    data-driven tree structure optimization for tensor network ML.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        bond_dim: int = 16,
        feature_map_config: Optional[dict] = None,
        init_method: str = "qr",
        initial_temperature: float = 1.0,
        anneal_rate: float = 0.003,
        mi_init: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bond_dim = bond_dim
        self.temperature = initial_temperature
        self.anneal_rate = anneal_rate
        self.mi_init = mi_init

        # Feature map
        if feature_map_config is None:
            feature_map_config = {"type": "trigonometric", "local_dim": 2}
        self.feature_map = get_feature_map(feature_map_config)
        self.local_dim = feature_map_config.get("local_dim", 2)

        # Build tree structure
        _, self.num_padded = build_binary_tree_structure(input_dim)
        self.num_tree_layers = int(math.log2(self.num_padded))

        # Build adaptive TTN layers
        self.ttn_layers = nn.ModuleList()
        d_in = self.local_dim
        current_size = self.num_padded

        for layer_idx in range(self.num_tree_layers):
            num_nodes = current_size // 2

            layer = AdaptiveTTNLayer(
                num_nodes=num_nodes,
                num_inputs=current_size,
                d_left=d_in,
                d_right=d_in,
                chi_out=bond_dim,
                adaptive=(current_size > 2),  # only adaptive for layers with >2 inputs
                temperature=initial_temperature,
                init_method=init_method,
            )
            self.ttn_layers.append(layer)
            d_in = bond_dim
            current_size = num_nodes

        # Classification head
        self.classifier = nn.Linear(bond_dim, num_classes)

        # Track temperature for annealing
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def anneal_temperature(self) -> float:
        """Anneal Gumbel-Softmax temperature. Call once per epoch."""
        self._step += 1
        self.temperature = max(
            0.1,  # minimum temperature
            self.temperature * math.exp(-self.anneal_rate)
        )
        # Update all pairing layers
        for layer in self.ttn_layers:
            if layer.pairing is not None:
                layer.pairing.temperature = self.temperature

        return self.temperature

    def initialize_from_mi(self, data: torch.Tensor) -> None:
        """
        Initialize feature pairing using mutual information.

        Computes pairwise MI between features and initializes the
        Gumbel-Softmax logits to prefer high-MI pairings.

        Args:
            data: (num_samples, input_dim) — subset of training data
        """
        print("Computing mutual information for topology initialization...")
        mi_matrix = compute_mutual_information(data[:5000])  # use subset for speed

        # Get MI-guided pairing
        mi_layers = mi_guided_pairing(mi_matrix)

        # Initialize assignment logits to prefer MI-guided pairs
        for layer_idx, layer in enumerate(self.ttn_layers):
            if layer.pairing is not None and layer_idx < len(mi_layers):
                mi_pairs = mi_layers[layer_idx]
                for pair_idx, (left, right) in enumerate(mi_pairs):
                    if pair_idx < layer.pairing.num_pairs:
                        # Set high logit for MI-preferred features
                        with torch.no_grad():
                            layer.pairing.assignment_logits[pair_idx, 0] *= 0.01
                            layer.pairing.assignment_logits[pair_idx, 1] *= 0.01
                            if left < layer.pairing.num_features:
                                layer.pairing.assignment_logits[pair_idx, 0, left] = 3.0
                            if right < layer.pairing.num_features:
                                layer.pairing.assignment_logits[pair_idx, 1, right] = 3.0

        print("MI-guided initialization complete.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive topology."""
        batch_size = x.shape[0]

        # Feature map
        features = self.feature_map(x)
        feature_list = [features[:, i, :] for i in range(features.shape[1])]

        # Pad
        while len(feature_list) < self.num_padded:
            pad = torch.ones(batch_size, self.local_dim, device=x.device) / math.sqrt(self.local_dim)
            feature_list.append(pad)

        # Contract through adaptive tree
        current = feature_list
        use_hard = not self.training  # hard assignments at inference
        for layer in self.ttn_layers:
            current = layer(current, hard=use_hard)

        # Classify
        root = current[0]
        logits = self.classifier(root)
        return logits

    def get_learned_topology(self) -> List[List[Tuple[int, int]]]:
        """Get the current hard-assignment topology."""
        topology = []
        for layer in self.ttn_layers:
            if layer.pairing is not None:
                topology.append(layer.pairing.get_hard_assignments())
            else:
                topology.append([(0, 1)])
        return topology

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_classes={self.num_classes}, "
            f"bond_dim={self.bond_dim}, temperature={self.temperature:.3f}, "
            f"num_layers={self.num_tree_layers}"
        )
