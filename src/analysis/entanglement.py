"""
Entanglement Entropy Analysis for TTN Interpretability.

NOVEL CONTRIBUTION: Uses quantum information-theoretic measures to provide
physics-grounded interpretability for the TTN classifier.

For a TTN with state |Ψ⟩, the entanglement entropy across any bond
measures the complexity of correlations that flow through that bond:

    S(bond) = -Tr(ρ_A log₂ ρ_A)

where ρ_A = Tr_B(|Ψ⟩⟨Ψ|) is the reduced density matrix obtained by
tracing out one side of the bond.

High entropy → features on both sides of the bond are strongly correlated
Low entropy → features are weakly correlated, bond could be compressed

This provides:
1. Feature importance ranking (which features contribute most to classification)
2. Layer-wise entanglement profiles (how information flows through the tree)
3. Optimal bond dimension selection (entropy sets the natural compression limit)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple


def compute_bond_entropy(tensor: torch.Tensor, bond_idx: int = -1) -> float:
    """
    Compute entanglement entropy across a bond of a TTN node tensor.

    For a tensor T of shape (d₁, d₂, χ), the entropy across the
    (d₁d₂, χ) bipartition is computed via SVD.

    Args:
        tensor: Node tensor of shape (d_left, d_right, chi)
        bond_idx: Which bond to compute entropy for (-1 = output bond)

    Returns:
        Von Neumann entropy in bits (log₂)
    """
    shape = tensor.shape

    if bond_idx == -1:
        # Bipartition: (d_left * d_right) vs (chi)
        mat = tensor.reshape(-1, shape[-1])
    elif bond_idx == 0:
        # Bipartition: (d_left) vs (d_right * chi)
        mat = tensor.reshape(shape[0], -1)
    else:
        # Bipartition: (d_left * ... ) vs (... * chi)
        left_size = 1
        for i in range(bond_idx + 1):
            left_size *= shape[i]
        mat = tensor.reshape(left_size, -1)

    # SVD to get Schmidt values
    _, S, _ = torch.linalg.svd(mat, full_matrices=False)

    # Normalize Schmidt values
    S = S / (S.norm() + 1e-12)

    # Square to get eigenvalues of reduced density matrix
    p = S ** 2
    p = p[p > 1e-12]  # remove zeros

    # Von Neumann entropy: S = -Σ p_i log₂(p_i)
    entropy = -(p * torch.log2(p)).sum().item()

    return entropy


def compute_bond_entanglement(model: nn.Module) -> List[List[float]]:
    """
    Compute entanglement entropy for every bond in the TTN.

    Args:
        model: TTN model (TreeTensorNetwork or AugmentedTTN)

    Returns:
        List of layers, each containing entropies for each node in that layer
    """
    layer_entropies = []

    # Get node tensors
    if hasattr(model, "ttn_layers"):
        for layer in model.ttn_layers:
            node_entropies = []
            nodes = layer.nodes if hasattr(layer, "nodes") else layer
            if isinstance(nodes, nn.ParameterList):
                for node in nodes:
                    entropy = compute_bond_entropy(node.data)
                    node_entropies.append(entropy)
            layer_entropies.append(node_entropies)
    elif hasattr(model, "attn_layers"):
        for layer in model.attn_layers:
            node_entropies = []
            if hasattr(layer, "nodes"):
                for node in layer.nodes:
                    entropy = compute_bond_entropy(node.data)
                    node_entropies.append(entropy)
            layer_entropies.append(node_entropies)

    return layer_entropies


@torch.no_grad()
def compute_feature_entanglement_map(
    model: nn.Module,
    data: torch.Tensor,
    num_samples: int = 500,
) -> np.ndarray:
    """
    Compute pairwise entanglement between features using intermediate TTN states.

    For each pair of features (i, j), compute the mutual information
    in the TTN representation by analyzing the reduced density matrices.

    This produces an N×N matrix where entry (i,j) quantifies how
    "entangled" features i and j are in the learned TTN representation.

    Args:
        model: Trained TTN model
        data: (num_samples, input_dim) data tensor
        num_samples: Number of samples to use

    Returns:
        (input_dim, input_dim) numpy array of pairwise entanglement values
    """
    model.eval()
    data = data[:num_samples]

    # Get intermediate states
    if not hasattr(model, "get_intermediate_states"):
        raise ValueError("Model must have get_intermediate_states() method")

    states = model.get_intermediate_states(data)

    # Layer 0: individual features (B, N, d)
    leaf_states = states[0]  # List of (B, d) tensors
    N = len(leaf_states)

    entanglement_map = np.zeros((N, N))

    # For each pair, compute correlation in the embedding space
    for i in range(N):
        for j in range(i + 1, N):
            # Compute joint density matrix from embeddings
            phi_i = leaf_states[i]  # (B, d)
            phi_j = leaf_states[j]  # (B, d)

            # Outer product averaged over batch: ρ_ij = E[φᵢ ⊗ φⱼ]
            rho_joint = torch.einsum("bi,bj->ij", phi_i, phi_j) / data.shape[0]

            # SVD to get Schmidt decomposition
            _, S, _ = torch.linalg.svd(rho_joint, full_matrices=False)
            S = S / (S.norm() + 1e-12)
            p = S ** 2
            p = p[p > 1e-12]

            # Entropy
            entropy = -(p * torch.log2(p)).sum().item()
            entanglement_map[i, j] = entropy
            entanglement_map[j, i] = entropy

    return entanglement_map


def layer_wise_entanglement_profile(model: nn.Module) -> Dict[str, List[float]]:
    """
    Compute statistics of entanglement entropy at each layer.

    Returns a profile showing how entanglement changes as information
    flows up the tree. In a well-trained TTN:
    - Lower layers: moderate entropy (local features)
    - Middle layers: peak entropy (feature integration)
    - Upper layers: may decrease (task-relevant compression)

    Args:
        model: Trained TTN model

    Returns:
        Dict with 'mean', 'max', 'min', 'std' entropy per layer
    """
    bond_entropies = compute_bond_entanglement(model)

    profile = {
        "layer": [],
        "mean_entropy": [],
        "max_entropy": [],
        "min_entropy": [],
        "std_entropy": [],
        "num_nodes": [],
    }

    for layer_idx, entropies in enumerate(bond_entropies):
        if not entropies:
            continue
        profile["layer"].append(layer_idx)
        profile["mean_entropy"].append(float(np.mean(entropies)))
        profile["max_entropy"].append(float(np.max(entropies)))
        profile["min_entropy"].append(float(np.min(entropies)))
        profile["std_entropy"].append(float(np.std(entropies)))
        profile["num_nodes"].append(len(entropies))

    return profile
