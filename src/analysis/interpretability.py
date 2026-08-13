"""
Interpretability analysis using entanglement entropy.

Provides feature importance ranking and class-conditional analysis
based on the quantum information-theoretic properties of the TTN.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.analysis.entanglement import compute_feature_entanglement_map


def feature_importance_from_entanglement(
    model: nn.Module,
    data: torch.Tensor,
    num_samples: int = 500,
) -> Dict[str, np.ndarray]:
    """
    Rank features by importance using entanglement entropy.

    A feature is "important" if it has high entanglement with many
    other features — meaning it carries information that is strongly
    correlated with the rest of the input.

    Args:
        model: Trained TTN model
        data: Input data tensor

    Returns:
        Dict with:
        - 'importance_scores': (N,) importance score per feature
        - 'ranking': (N,) feature indices sorted by importance (descending)
        - 'entanglement_map': (N, N) pairwise entanglement matrix
    """
    ent_map = compute_feature_entanglement_map(model, data, num_samples)

    # Feature importance = total entanglement with all other features
    importance = ent_map.sum(axis=1)

    # Normalize to [0, 1]
    if importance.max() > 0:
        importance = importance / importance.max()

    ranking = np.argsort(importance)[::-1]

    return {
        "importance_scores": importance,
        "ranking": ranking,
        "entanglement_map": ent_map,
    }


@torch.no_grad()
def class_conditional_entanglement(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
    num_samples_per_class: int = 100,
) -> Dict[int, np.ndarray]:
    """
    Compute entanglement maps conditioned on class labels.

    Different classes may exhibit different entanglement patterns,
    revealing which feature correlations are class-specific.

    For example, in MNIST:
    - Digit '0' may show high entanglement between border pixels
    - Digit '1' may show entanglement concentrated along a vertical strip

    Args:
        model: Trained TTN model
        data: (N, input_dim) full dataset
        labels: (N,) class labels
        num_classes: Number of classes
        num_samples_per_class: Samples per class for estimation

    Returns:
        Dict mapping class_id → (input_dim, input_dim) entanglement map
    """
    class_maps = {}

    for c in range(num_classes):
        mask = labels == c
        class_data = data[mask]

        if len(class_data) < 10:
            continue

        class_data = class_data[:num_samples_per_class]

        try:
            ent_map = compute_feature_entanglement_map(model, class_data, num_samples_per_class)
            class_maps[c] = ent_map
        except Exception as e:
            print(f"  Warning: Could not compute entanglement for class {c}: {e}")

    return class_maps


def entanglement_pruning_analysis(
    model: nn.Module,
    data: torch.Tensor,
    prune_fractions: List[float] = [0.1, 0.2, 0.3, 0.5, 0.7],
) -> Dict[str, List]:
    """
    Analyze how removing low-entanglement features affects representations.

    This reveals which features are redundant and could be safely pruned
    without losing classification performance.

    Args:
        model: Trained TTN model
        data: Input data
        prune_fractions: Fractions of features to prune

    Returns:
        Dict with pruning results
    """
    importance = feature_importance_from_entanglement(model, data)
    ranking = importance["ranking"]
    scores = importance["importance_scores"]

    N = len(ranking)
    results = {
        "prune_fraction": [],
        "features_kept": [],
        "total_importance_kept": [],
        "lowest_kept_importance": [],
    }

    for frac in prune_fractions:
        num_prune = int(N * frac)
        num_keep = N - num_prune

        kept_features = ranking[:num_keep]
        total_importance = scores[kept_features].sum()

        results["prune_fraction"].append(frac)
        results["features_kept"].append(num_keep)
        results["total_importance_kept"].append(float(total_importance))
        results["lowest_kept_importance"].append(float(scores[kept_features[-1]]) if num_keep > 0 else 0.0)

    return results
