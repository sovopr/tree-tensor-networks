"""
Dynamic Bond Dimension Tree Tensor Network.

NOVEL CONTRIBUTION (Tanya Mittal): Instead of a fixed bond dimension chi
across all layers, this model allows each TTN layer/bond to select its
own effective chi from a discrete set {2, 4, 8, 16} using a learnable
Gumbel-Softmax selection mechanism.

A complexity/parameter penalty encourages the model to use smaller bond
dimensions where possible, achieving automatic compression without
manual tuning.

Variants:
    - DynamicBondTTN: Fixed topology, learnable chi per layer
    - FullyAdaptiveTTN: Learnable topology + learnable chi per layer

The key insight is that lower layers of the tree (processing raw pixel
pairs) often don't need large bond dimensions, while upper layers
(integrating complex features) may benefit from higher capacity. This
model discovers the optimal per-layer bond dimensions automatically.
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


class BondDimensionSelector(nn.Module):
    """
    Learnable bond dimension selector using Gumbel-Softmax.

    For a given layer, maintains tensors for each candidate bond dimension
    and a set of logits that determine which bond dimension to use.
    During training, uses Gumbel-Softmax to softly mix outputs from
    all candidate dimensions. At inference, selects the best one.

    Args:
        candidate_dims: List of candidate bond dimensions (e.g., [2, 4, 8, 16])
        temperature: Gumbel-Softmax temperature
    """

    def __init__(
        self,
        candidate_dims: List[int],
        temperature: float = 1.0,
    ):
        super().__init__()
        self.candidate_dims = candidate_dims
        self.num_candidates = len(candidate_dims)
        self.temperature = temperature

        # Learnable logits for bond dimension selection
        # Initialize with slight preference for middle dimensions
        init_logits = torch.zeros(self.num_candidates)
        mid = self.num_candidates // 2
        init_logits[mid] = 0.5  # slight bias toward middle
        self.selection_logits = nn.Parameter(init_logits)

    def get_weights(self, hard: bool = False) -> torch.Tensor:
        """
        Get selection weights via Gumbel-Softmax.

        Args:
            hard: If True, use hard argmax (for inference)

        Returns:
            (num_candidates,) weight vector that sums to 1
        """
        if hard or not self.training:
            # Hard selection: one-hot
            idx = self.selection_logits.argmax()
            weights = torch.zeros_like(self.selection_logits)
            weights[idx] = 1.0
            return weights
        else:
            return F.gumbel_softmax(
                self.selection_logits.unsqueeze(0),
                tau=self.temperature,
                hard=False,
                dim=-1,
            ).squeeze(0)

    def get_selected_dim(self) -> int:
        """Get the currently selected bond dimension (hard argmax)."""
        idx = self.selection_logits.argmax().item()
        return self.candidate_dims[idx]

    def complexity_penalty(self) -> torch.Tensor:
        """
        Compute a differentiable complexity penalty based on expected bond dimension.

        The penalty is proportional to the expected bond dimension, encouraging
        the model to prefer smaller dimensions when accuracy is comparable.

        Returns:
            Scalar penalty value
        """
        weights = F.softmax(self.selection_logits, dim=0)
        dims_tensor = torch.tensor(
            self.candidate_dims, dtype=torch.float32,
            device=self.selection_logits.device,
        )
        # Expected dimension, normalized by max dimension
        expected_dim = (weights * dims_tensor).sum()
        return expected_dim / max(self.candidate_dims)


class DynamicBondTTNLayer(nn.Module):
    """
    TTN layer with learnable bond dimension.

    Maintains separate node tensors for each candidate bond dimension.
    During forward pass, computes outputs for all candidates and blends
    them according to the learned selection weights.
    """

    def __init__(
        self,
        num_nodes: int,
        d_left: int,
        d_right: int,
        candidate_dims: List[int],
        temperature: float = 1.0,
        init_method: str = "qr",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.candidate_dims = candidate_dims
        self.max_dim = max(candidate_dims)

        # Bond dimension selector for this layer
        self.bond_selector = BondDimensionSelector(candidate_dims, temperature)

        # Create node tensors for EACH candidate bond dimension
        # We use the max dimension and mask/truncate for smaller ones
        self.nodes = nn.ParameterList()
        for _ in range(num_nodes):
            if init_method == "qr":
                tensor = qr_init((d_left, d_right, self.max_dim))
            else:
                tensor = torch.randn(d_left, d_right, self.max_dim) * 0.01
            self.nodes.append(nn.Parameter(tensor))

    def forward(
        self,
        inputs: List[torch.Tensor],
        hard: bool = False,
    ) -> List[torch.Tensor]:
        """
        Contract with dynamically selected bond dimension.

        During training: compute full-rank contraction, then project
        down to a weighted blend of different truncation levels.
        During inference: use only the selected bond dimension.
        """
        weights = self.bond_selector.get_weights(hard=hard)

        outputs = []
        for i in range(self.num_nodes):
            left_idx = 2 * i
            right_idx = 2 * i + 1
            left = inputs[left_idx]
            right = inputs[right_idx] if right_idx < len(inputs) else inputs[left_idx]

            # Full contraction with max bond dim
            full_output = contract_pair(left, right, self.nodes[i])  # (B, max_dim)

            if hard or not self.training:
                # At inference: truncate to selected dimension, then pad back
                selected_dim = self.bond_selector.get_selected_dim()
                truncated = full_output[:, :selected_dim]
                # Pad to max_dim for consistent downstream dimensions
                if selected_dim < self.max_dim:
                    padding = torch.zeros(
                        full_output.shape[0], self.max_dim - selected_dim,
                        device=full_output.device, dtype=full_output.dtype,
                    )
                    output = torch.cat([truncated, padding], dim=-1)
                else:
                    output = truncated
            else:
                # During training: blend truncated outputs
                # Each candidate gets a weight; the output is a soft blend
                blended = torch.zeros_like(full_output)
                for c_idx, chi in enumerate(self.candidate_dims):
                    # Mask: keep first chi dimensions, zero the rest
                    mask = torch.zeros(self.max_dim, device=full_output.device)
                    mask[:chi] = 1.0
                    blended = blended + weights[c_idx] * (full_output * mask)
                output = blended

            outputs.append(output)

        return outputs


class DynamicBondTTN(nn.Module):
    """
    Tree Tensor Network with learnable per-layer bond dimensions.

    Each layer independently learns its optimal bond dimension from a
    discrete candidate set. A complexity penalty in the loss function
    encourages smaller dimensions where possible.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        candidate_dims: List of candidate bond dimensions per layer
        feature_map_config: Feature map configuration
        init_method: Tensor initialization method
        initial_temperature: Gumbel-Softmax temperature
        anneal_rate: Temperature annealing rate
        complexity_weight: Weight for the complexity penalty in loss
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        candidate_dims: Optional[List[int]] = None,
        feature_map_config: Optional[dict] = None,
        init_method: str = "qr",
        initial_temperature: float = 1.0,
        anneal_rate: float = 0.003,
        complexity_weight: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temperature = initial_temperature
        self.anneal_rate = anneal_rate
        self.complexity_weight = complexity_weight

        if candidate_dims is None:
            candidate_dims = [2, 4, 8, 16]
        self.candidate_dims = candidate_dims
        self.max_dim = max(candidate_dims)

        # Feature map
        if feature_map_config is None:
            feature_map_config = {"type": "trigonometric", "local_dim": 2}
        self.feature_map = get_feature_map(feature_map_config)
        self.local_dim = feature_map_config.get("local_dim", 2)

        # Build tree structure
        self.tree_layers_structure, self.num_padded = build_binary_tree_structure(input_dim)
        self.num_tree_layers = len(self.tree_layers_structure)

        # Build dynamic TTN layers
        self.ttn_layers = nn.ModuleList()
        d_in = self.local_dim

        for layer_idx in range(self.num_tree_layers):
            num_nodes = len(self.tree_layers_structure[layer_idx])

            layer = DynamicBondTTNLayer(
                num_nodes=num_nodes,
                d_left=d_in,
                d_right=d_in,
                candidate_dims=candidate_dims,
                temperature=initial_temperature,
                init_method=init_method,
            )
            self.ttn_layers.append(layer)
            d_in = self.max_dim  # all layers output max_dim (padded)

        # Classification head (uses max_dim since outputs are padded)
        self.classifier = nn.Linear(self.max_dim, num_classes)

        # Track temperature for annealing
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def anneal_temperature(self) -> float:
        """Anneal Gumbel-Softmax temperature. Call once per epoch."""
        self._step += 1
        self.temperature = max(
            0.1,
            self.temperature * math.exp(-self.anneal_rate)
        )
        for layer in self.ttn_layers:
            layer.bond_selector.temperature = self.temperature
        return self.temperature

    def get_complexity_penalty(self) -> torch.Tensor:
        """
        Total complexity penalty across all layers.

        This should be added to the classification loss during training:
            total_loss = ce_loss + model.complexity_weight * model.get_complexity_penalty()
        """
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.ttn_layers:
            penalty = penalty + layer.bond_selector.complexity_penalty()
        return penalty / self.num_tree_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic bond dimensions."""
        batch_size = x.shape[0]

        # Feature map
        features = self.feature_map(x)
        feature_list = [features[:, i, :] for i in range(features.shape[1])]

        # Pad
        while len(feature_list) < self.num_padded:
            pad = torch.ones(batch_size, self.local_dim, device=x.device) / math.sqrt(self.local_dim)
            feature_list.append(pad)

        # Contract through dynamic tree
        current = feature_list
        use_hard = not self.training
        for layer in self.ttn_layers:
            current = layer(current, hard=use_hard)

        # Classify
        root = current[0]
        logits = self.classifier(root)
        return logits

    def get_selected_dimensions(self) -> List[int]:
        """Get the currently selected bond dimension for each layer."""
        return [layer.bond_selector.get_selected_dim() for layer in self.ttn_layers]

    def extra_repr(self) -> str:
        selected = self.get_selected_dimensions()
        return (
            f"input_dim={self.input_dim}, num_classes={self.num_classes}, "
            f"candidate_dims={self.candidate_dims}, "
            f"selected_dims={selected}, "
            f"temperature={self.temperature:.3f}"
        )


class FullyAdaptiveTTN(nn.Module):
    """
    Fully Adaptive TTN: learnable topology + learnable bond dimensions.

    Combines the Adaptive TTN's Gumbel-Softmax topology learning with
    the Dynamic Bond TTN's per-layer bond dimension selection. This is
    the most flexible variant, simultaneously optimizing both the tree
    structure and the compression level at each layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        candidate_dims: Optional[List[int]] = None,
        feature_map_config: Optional[dict] = None,
        init_method: str = "qr",
        initial_temperature: float = 1.0,
        anneal_rate: float = 0.003,
        complexity_weight: float = 0.01,
        mi_init: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temperature = initial_temperature
        self.anneal_rate = anneal_rate
        self.complexity_weight = complexity_weight
        self.mi_init = mi_init

        if candidate_dims is None:
            candidate_dims = [2, 4, 8, 16]
        self.candidate_dims = candidate_dims
        self.max_dim = max(candidate_dims)

        # Feature map
        if feature_map_config is None:
            feature_map_config = {"type": "trigonometric", "local_dim": 2}
        self.feature_map = get_feature_map(feature_map_config)
        self.local_dim = feature_map_config.get("local_dim", 2)

        # Build tree structure
        _, self.num_padded = build_binary_tree_structure(input_dim)
        self.num_tree_layers = int(math.log2(self.num_padded))

        # Import adaptive pairing from adaptive_ttn
        from src.models.adaptive_ttn import GumbelSoftmaxPairing

        # Build fully adaptive layers (topology + bond dim)
        self.ttn_layers = nn.ModuleList()
        self.pairings = nn.ModuleList()
        d_in = self.local_dim
        current_size = self.num_padded

        for layer_idx in range(self.num_tree_layers):
            num_nodes = current_size // 2

            # Adaptive pairing (topology)
            if current_size > 2:
                pairing = GumbelSoftmaxPairing(current_size, initial_temperature)
            else:
                pairing = None
            self.pairings.append(pairing)

            # Dynamic bond dimension layer
            layer = DynamicBondTTNLayer(
                num_nodes=num_nodes,
                d_left=d_in,
                d_right=d_in,
                candidate_dims=candidate_dims,
                temperature=initial_temperature,
                init_method=init_method,
            )
            self.ttn_layers.append(layer)
            d_in = self.max_dim
            current_size = num_nodes

        # Classification head
        self.classifier = nn.Linear(self.max_dim, num_classes)
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def anneal_temperature(self) -> float:
        """Anneal temperature for both topology and bond selection."""
        self._step += 1
        self.temperature = max(0.1, self.temperature * math.exp(-self.anneal_rate))
        for layer in self.ttn_layers:
            layer.bond_selector.temperature = self.temperature
        for pairing in self.pairings:
            if pairing is not None:
                pairing.temperature = self.temperature
        return self.temperature

    def initialize_from_mi(self, data: torch.Tensor) -> None:
        """Initialize topology from mutual information (same as AdaptiveTTN)."""
        print("Computing mutual information for topology initialization...")
        mi_matrix = compute_mutual_information(data[:5000])
        mi_layers = mi_guided_pairing(mi_matrix)

        for layer_idx, pairing in enumerate(self.pairings):
            if pairing is not None and layer_idx < len(mi_layers):
                mi_pairs = mi_layers[layer_idx]
                for pair_idx, (left, right) in enumerate(mi_pairs):
                    if pair_idx < pairing.num_pairs:
                        with torch.no_grad():
                            pairing.assignment_logits[pair_idx, 0] *= 0.01
                            pairing.assignment_logits[pair_idx, 1] *= 0.01
                            if left < pairing.num_features:
                                pairing.assignment_logits[pair_idx, 0, left] = 3.0
                            if right < pairing.num_features:
                                pairing.assignment_logits[pair_idx, 1, right] = 3.0

        print("MI-guided initialization complete.")

    def get_complexity_penalty(self) -> torch.Tensor:
        """Total complexity penalty across all layers."""
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.ttn_layers:
            penalty = penalty + layer.bond_selector.complexity_penalty()
        return penalty / self.num_tree_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive topology and dynamic bond dimensions."""
        batch_size = x.shape[0]

        # Feature map
        features = self.feature_map(x)
        feature_list = [features[:, i, :] for i in range(features.shape[1])]

        # Pad
        while len(feature_list) < self.num_padded:
            pad = torch.ones(batch_size, self.local_dim, device=x.device) / math.sqrt(self.local_dim)
            feature_list.append(pad)

        # Contract through fully adaptive tree
        current = feature_list
        use_hard = not self.training

        for layer_idx, (layer, pairing) in enumerate(zip(self.ttn_layers, self.pairings)):
            if pairing is not None:
                # Adaptive pairing
                left_list, right_list = pairing(current, hard=use_hard)
                # Contract paired features
                outputs = []
                for i in range(layer.num_nodes):
                    output = contract_pair(left_list[i], right_list[i], layer.nodes[i])
                    # Apply bond dimension selection
                    weights = layer.bond_selector.get_weights(hard=use_hard)
                    if use_hard:
                        selected_dim = layer.bond_selector.get_selected_dim()
                        output[:, selected_dim:] = 0.0
                    else:
                        blended = torch.zeros_like(output)
                        for c_idx, chi in enumerate(layer.candidate_dims):
                            mask = torch.zeros(layer.max_dim, device=output.device)
                            mask[:chi] = 1.0
                            blended = blended + weights[c_idx] * (output * mask)
                        output = blended
                    outputs.append(output)
                current = outputs
            else:
                # Fixed pairing for small layers
                current = layer(current, hard=use_hard)

        # Classify
        root = current[0]
        logits = self.classifier(root)
        return logits

    def get_selected_dimensions(self) -> List[int]:
        """Get selected bond dimension for each layer."""
        return [layer.bond_selector.get_selected_dim() for layer in self.ttn_layers]

    def get_learned_topology(self) -> List[List[Tuple[int, int]]]:
        """Get the current hard-assignment topology."""
        topology = []
        for pairing in self.pairings:
            if pairing is not None:
                topology.append(pairing.get_hard_assignments())
            else:
                topology.append([(0, 1)])
        return topology

    def extra_repr(self) -> str:
        selected = self.get_selected_dimensions()
        return (
            f"input_dim={self.input_dim}, num_classes={self.num_classes}, "
            f"candidate_dims={self.candidate_dims}, "
            f"selected_dims={selected}, "
            f"temperature={self.temperature:.3f}"
        )
