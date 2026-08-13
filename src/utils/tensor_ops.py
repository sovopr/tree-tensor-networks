"""
Utility functions for tensor operations in Tree Tensor Networks.

Includes:
- Isometric (QR) initialization for stable training
- Pairwise tensor contraction
- SVD-based bond dimension truncation
- Tree topology construction
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional


def qr_init(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Initialize a tensor with QR decomposition for isometric property.

    For a tensor of shape (d1, d2, ..., dk, chi), we reshape it to
    (d1*d2*...*dk, chi), apply QR, and reshape back. This ensures
    the tensor is (approximately) isometric, which stabilizes training.
    """
    total_in = 1
    for s in shape[:-1]:
        total_in *= s
    out_dim = shape[-1]

    # Random matrix
    mat = torch.randn(total_in, out_dim, dtype=dtype)

    # QR decomposition
    if total_in >= out_dim:
        q, _ = torch.linalg.qr(mat)
        q = q[:, :out_dim]
    else:
        q, _ = torch.linalg.qr(mat.T)
        q = q[:, :total_in].T

    return q.reshape(shape)


def random_init(
    shape: Tuple[int, ...],
    std: float = 0.01,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Initialize a tensor with small random values."""
    return torch.randn(shape, dtype=dtype) * std


def contract_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    node_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Contract a pair of local feature vectors with a TTN node tensor.

    Args:
        left:  (batch, d_left)  — left child feature/output
        right: (batch, d_right) — right child feature/output
        node_tensor: (d_left, d_right, chi_out) — the TTN isometry

    Returns:
        (batch, chi_out) — output of this TTN node
    """
    # Einstein summation: batch contraction of left and right with the node
    # left_i * right_j * node_{i,j,k} -> output_k
    return torch.einsum("bi,bj,ijk->bk", left, right, node_tensor)


def contract_pair_batched(
    left: torch.Tensor,
    right: torch.Tensor,
    node_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Same as contract_pair but optimized for larger batches.
    Uses two sequential matrix multiplications instead of einsum.

    Args:
        left:  (batch, d_left)
        right: (batch, d_right)
        node_tensor: (d_left, d_right, chi_out)

    Returns:
        (batch, chi_out)
    """
    d_left, d_right, chi_out = node_tensor.shape
    batch = left.shape[0]

    # Step 1: Contract left with node -> (batch, d_right, chi_out)
    # node reshaped: (d_left, d_right * chi_out)
    node_reshaped = node_tensor.reshape(d_left, d_right * chi_out)
    intermediate = left @ node_reshaped  # (batch, d_right * chi_out)
    intermediate = intermediate.reshape(batch, d_right, chi_out)

    # Step 2: Contract with right -> (batch, chi_out)
    # right: (batch, d_right) -> (batch, 1, d_right)
    output = torch.bmm(right.unsqueeze(1), intermediate).squeeze(1)  # (batch, chi_out)

    return output


def build_binary_tree_structure(num_leaves: int) -> List[List[Tuple[int, int]]]:
    """
    Build a balanced binary tree structure for TTN.

    Given N leaves (features), creates log2(N) layers of pairwise contractions.
    If N is not a power of 2, pads with identity connections.

    Args:
        num_leaves: Number of input features

    Returns:
        List of layers, where each layer is a list of (left_idx, right_idx) pairs
        referring to the outputs of the previous layer.
    """
    # Pad to next power of 2
    num_padded = 2 ** math.ceil(math.log2(max(num_leaves, 2)))

    layers = []
    current_count = num_padded

    while current_count > 1:
        layer = []
        for i in range(0, current_count, 2):
            layer.append((i, i + 1))
        layers.append(layer)
        current_count = current_count // 2

    return layers, num_padded


def truncated_svd(
    tensor: torch.Tensor,
    max_rank: int,
    dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform truncated SVD on a tensor (reshaped as matrix).

    Args:
        tensor: Input tensor
        max_rank: Maximum rank to keep
        dim: Dimension to split on for matricization

    Returns:
        (U, S, Vh) with rank truncated to max_rank
    """
    # Reshape to matrix
    shape = tensor.shape
    left_dims = shape[:dim + 1]
    right_dims = shape[dim + 1:]
    left_size = 1
    for d in left_dims:
        left_size *= d
    right_size = 1
    for d in right_dims:
        right_size *= d

    mat = tensor.reshape(left_size, right_size)

    # SVD
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

    # Truncate
    rank = min(max_rank, len(S))
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    return U, S, Vh


def compute_mutual_information(
    data: torch.Tensor,
    num_bins: int = 20,
) -> torch.Tensor:
    """
    Compute pairwise mutual information between all features.

    Uses histogram-based estimation. For N features, returns an NxN
    symmetric matrix where MI[i,j] measures the statistical dependence
    between features i and j.

    This is used by the Adaptive TTN to determine optimal feature pairing.

    Args:
        data: (num_samples, num_features) tensor
        num_bins: Number of bins for histogram estimation

    Returns:
        (num_features, num_features) mutual information matrix
    """
    N = data.shape[1]
    mi_matrix = torch.zeros(N, N)

    # Discretize features into bins
    data_np = data.cpu().numpy()
    bins = num_bins

    for i in range(N):
        for j in range(i + 1, N):
            # Joint histogram
            hist_2d, _, _ = torch.histogram(
                torch.stack([data[:, i], data[:, j]], dim=0).T.contiguous(),
                bins=bins,
            ) if hasattr(torch, 'histogram') else (None, None, None)

            # Fall back to numpy if torch.histogram doesn't support 2D
            import numpy as np
            hist_2d_np, _, _ = np.histogram2d(
                data_np[:, i], data_np[:, j], bins=bins
            )

            # Normalize to joint probability
            p_joint = hist_2d_np / hist_2d_np.sum()
            p_joint = p_joint + 1e-10  # avoid log(0)

            # Marginals
            p_i = p_joint.sum(axis=1)
            p_j = p_joint.sum(axis=0)

            # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
            mi = 0.0
            for a in range(bins):
                for b in range(bins):
                    if p_joint[a, b] > 1e-10:
                        mi += p_joint[a, b] * np.log(
                            p_joint[a, b] / (p_i[a] * p_j[b] + 1e-10)
                        )

            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def mi_guided_pairing(mi_matrix: torch.Tensor) -> List[List[Tuple[int, int]]]:
    """
    Build tree structure guided by mutual information.

    At each level, pair features with highest mutual information first.
    Uses greedy matching on the MI matrix.

    Args:
        mi_matrix: (N, N) mutual information matrix

    Returns:
        List of layers with pairing indices
    """
    N = mi_matrix.shape[0]

    # Pad to power of 2
    num_padded = 2 ** math.ceil(math.log2(max(N, 2)))
    if num_padded > N:
        # Extend MI matrix with zeros for padding
        extended = torch.zeros(num_padded, num_padded)
        extended[:N, :N] = mi_matrix
        mi_matrix = extended

    layers = []
    current_indices = list(range(num_padded))

    while len(current_indices) > 1:
        # Build MI submatrix for current indices
        n = len(current_indices)
        paired = [False] * n
        layer = []

        # Greedy matching: pair features with highest MI
        mi_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                mi_val = mi_matrix[current_indices[i], current_indices[j]].item()
                mi_pairs.append((mi_val, i, j))

        mi_pairs.sort(reverse=True)

        next_indices = []
        pair_count = 0

        for _, i, j in mi_pairs:
            if not paired[i] and not paired[j]:
                layer.append((i, j))
                paired[i] = True
                paired[j] = True
                next_indices.append(pair_count)
                pair_count += 1

            if pair_count == n // 2:
                break

        # Handle any unpaired (shouldn't happen with power-of-2 padding)
        for i in range(n):
            if not paired[i]:
                # Pair with itself (identity contraction)
                layer.append((i, i))
                next_indices.append(pair_count)
                pair_count += 1

        layers.append(layer)
        current_indices = list(range(pair_count))

    return layers
