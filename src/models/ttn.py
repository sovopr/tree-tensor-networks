"""
Tree Tensor Network (TTN) Classifier.

The core model: a hierarchical tensor network that contracts input features
bottom-up through a binary tree to produce classification logits.

Architecture:
    Input (B, N) → Feature Map (B, N, d) → Binary Tree Contraction → Root (B, χ) → Logits (B, C)

Mathematical structure:
    - N input features, each mapped to local dimension d
    - log₂(N) layers of pairwise tensor contractions
    - Each node: isometry tensor of shape (d_left, d_right, χ_out)
    - Bond dimension χ controls expressivity vs compression
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Any

from src.data.feature_maps import get_feature_map
from src.utils.tensor_ops import (
    qr_init,
    random_init,
    contract_pair,
    build_binary_tree_structure,
)


class TTNLayer(nn.Module):
    """
    A single layer of the Tree Tensor Network.

    Contains multiple node tensors that perform pairwise contraction
    of inputs from the previous layer.
    """

    def __init__(
        self,
        num_nodes: int,
        d_left: int,
        d_right: int,
        chi_out: int,
        init_method: str = "qr",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_left = d_left
        self.d_right = d_right
        self.chi_out = chi_out

        # Create node tensors as parameters
        self.nodes = nn.ParameterList()
        for _ in range(num_nodes):
            if init_method == "qr":
                tensor = qr_init((d_left, d_right, chi_out))
            elif init_method == "random":
                tensor = random_init((d_left, d_right, chi_out), std=0.01)
            else:
                tensor = torch.randn(d_left, d_right, chi_out) * 0.01

            self.nodes.append(nn.Parameter(tensor))

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Contract pairs of inputs through the node tensors.

        Args:
            inputs: List of tensors, each (batch, d_in)

        Returns:
            List of tensors, each (batch, chi_out)
        """
        outputs = []
        for i in range(self.num_nodes):
            left_idx = 2 * i
            right_idx = 2 * i + 1

            left = inputs[left_idx]
            right = inputs[right_idx] if right_idx < len(inputs) else inputs[left_idx]

            output = contract_pair(left, right, self.nodes[i])
            outputs.append(output)

        return outputs


class TreeTensorNetwork(nn.Module):
    """
    Full Tree Tensor Network classifier.

    Combines a feature map with a hierarchical tree of tensor contractions,
    followed by a linear classification head.

    The bond dimension χ is the primary hyperparameter controlling the
    accuracy-compression trade-off. Typical values:
        χ=2:   Very compressed, fast, lower accuracy
        χ=8:   Good balance for MNIST
        χ=16:  Good balance for Fashion-MNIST
        χ=32:  Higher capacity for CIFAR-10
        χ=64:  Near full-rank for complex tasks
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        bond_dim: int = 8,
        feature_map_config: Optional[dict] = None,
        init_method: str = "qr",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bond_dim = bond_dim

        # Feature map
        if feature_map_config is None:
            feature_map_config = {"type": "trigonometric", "local_dim": 2}
        self.feature_map = get_feature_map(feature_map_config)
        self.local_dim = feature_map_config.get("local_dim", 2)

        # Build tree structure
        self.tree_layers_structure, self.num_padded = build_binary_tree_structure(input_dim)
        self.num_tree_layers = len(self.tree_layers_structure)

        # Build TTN layers
        self.ttn_layers = nn.ModuleList()
        d_in = self.local_dim  # dimension from feature map

        for layer_idx in range(self.num_tree_layers):
            num_nodes = len(self.tree_layers_structure[layer_idx])
            chi_out = bond_dim

            layer = TTNLayer(
                num_nodes=num_nodes,
                d_left=d_in,
                d_right=d_in,
                chi_out=chi_out,
                init_method=init_method,
            )
            self.ttn_layers.append(layer)
            d_in = chi_out  # next layer input dim = this layer's output dim

        # Classification head: root tensor output → logits
        self.classifier = nn.Linear(bond_dim, num_classes)

        # Store bond entropies for analysis
        self._bond_entropies: Optional[List[List[float]]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TTN.

        Args:
            x: (batch_size, input_dim) — flattened input features in [0, 1]

        Returns:
            (batch_size, num_classes) — classification logits
        """
        batch_size = x.shape[0]

        # 1. Apply feature map: (B, N) → (B, N, d)
        features = self.feature_map(x)

        # 2. Convert to list of feature vectors: [(B, d), (B, d), ...]
        feature_list = [features[:, i, :] for i in range(features.shape[1])]

        # 3. Pad to power of 2 if needed
        while len(feature_list) < self.num_padded:
            # Pad with uniform vectors (no information)
            pad = torch.ones(batch_size, self.local_dim, device=x.device) / math.sqrt(self.local_dim)
            feature_list.append(pad)

        # 4. Contract through tree layers
        current = feature_list
        for layer in self.ttn_layers:
            current = layer(current)

        # 5. Root tensor → logits
        root = current[0]  # (B, bond_dim)
        logits = self.classifier(root)  # (B, num_classes)

        return logits

    def get_intermediate_states(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Get intermediate contraction results at each tree level.
        Used for entanglement entropy analysis.

        Returns:
            List of layers, each containing list of tensors at that level.
        """
        batch_size = x.shape[0]
        features = self.feature_map(x)
        feature_list = [features[:, i, :] for i in range(features.shape[1])]

        while len(feature_list) < self.num_padded:
            pad = torch.ones(batch_size, self.local_dim, device=x.device) / math.sqrt(self.local_dim)
            feature_list.append(pad)

        all_states = [feature_list]
        current = feature_list

        for layer in self.ttn_layers:
            current = layer(current)
            all_states.append(current)

        return all_states

    def get_node_tensors(self) -> List[List[torch.Tensor]]:
        """Get all node tensors organized by layer for analysis."""
        return [
            [node.data for node in layer.nodes]
            for layer in self.ttn_layers
        ]

    def get_tree_info(self) -> Dict[str, Any]:
        """Return tree structure information for visualization."""
        return {
            "input_dim": self.input_dim,
            "num_padded": self.num_padded,
            "num_layers": self.num_tree_layers,
            "bond_dim": self.bond_dim,
            "local_dim": self.local_dim,
            "tree_structure": self.tree_layers_structure,
            "nodes_per_layer": [len(layer) for layer in self.tree_layers_structure],
        }

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_classes={self.num_classes}, "
            f"bond_dim={self.bond_dim}, local_dim={self.local_dim}, "
            f"num_layers={self.num_tree_layers}, num_padded={self.num_padded}"
        )
