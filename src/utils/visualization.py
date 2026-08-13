"""
Visualization utilities for TTN research.

Includes training curves, TTN structure diagrams, entanglement maps,
and publication-quality figure generation.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Publication-quality defaults
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> None:
    """Plot training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r--", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, val_accs, "r--", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_ttn_structure(
    num_features: int,
    tree_layers: List[List[Tuple[int, int]]],
    bond_dims: Optional[List[int]] = None,
    entanglement_values: Optional[List[List[float]]] = None,
    save_path: Optional[str] = None,
    title: str = "Tree Tensor Network Structure",
) -> None:
    """
    Visualize TTN tree structure with optional entanglement entropy coloring.

    Draws a bottom-up tree diagram showing the hierarchical contraction.
    """
    fig, ax = plt.subplots(figsize=(max(14, num_features * 0.3), 6))

    num_layers = len(tree_layers)
    y_positions = np.linspace(0, 1, num_layers + 2)

    # Draw leaf nodes
    leaf_x = np.linspace(0, 1, num_features)
    for i, x in enumerate(leaf_x):
        ax.plot(x, y_positions[0], "ko", markersize=6)
        ax.text(x, y_positions[0] - 0.05, f"x{i}", ha="center", fontsize=7)

    # Draw tree layers
    prev_positions = list(leaf_x)
    for layer_idx, layer in enumerate(tree_layers):
        y = y_positions[layer_idx + 1]
        new_positions = []

        for pair_idx, (left, right) in enumerate(layer):
            if left < len(prev_positions) and right < len(prev_positions):
                x_left = prev_positions[left]
                x_right = prev_positions[right]
                x_mid = (x_left + x_right) / 2
                new_positions.append(x_mid)

                # Color by entanglement if available
                color = "steelblue"
                if entanglement_values and layer_idx < len(entanglement_values):
                    if pair_idx < len(entanglement_values[layer_idx]):
                        entropy = entanglement_values[layer_idx][pair_idx]
                        color = plt.cm.viridis(entropy / max(0.01, max(entanglement_values[layer_idx])))

                # Draw connections
                ax.plot([x_left, x_mid], [y_positions[layer_idx], y], "-", color="gray", linewidth=1)
                ax.plot([x_right, x_mid], [y_positions[layer_idx], y], "-", color="gray", linewidth=1)

                # Draw node
                ax.plot(x_mid, y, "o", color=color, markersize=10, markeredgecolor="black", markeredgewidth=0.5)

                # Bond dimension label
                if bond_dims and layer_idx < len(bond_dims):
                    ax.text(x_mid + 0.02, y + 0.02, f"χ={bond_dims[layer_idx]}",
                            fontsize=7, color="darkred")

        prev_positions = new_positions

    # Root
    if prev_positions:
        ax.plot(prev_positions[0], y_positions[-1], "r*", markersize=15)
        ax.text(prev_positions[0], y_positions[-1] + 0.04, "Root", ha="center",
                fontsize=10, fontweight="bold", color="darkred")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.15)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_entanglement_map(
    entanglement_matrix: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Feature Entanglement Map",
    figsize: Tuple[int, int] = (8, 7),
) -> None:
    """Plot entanglement entropy as a heatmap between features."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        entanglement_matrix,
        cmap="magma",
        ax=ax,
        square=True,
        cbar_kws={"label": "Entanglement Entropy"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Feature Index")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_parameter_comparison(
    model_names: List[str],
    param_counts: List[int],
    accuracies: List[float],
    save_path: Optional[str] = None,
    title: str = "Parameter Efficiency",
) -> None:
    """
    Scatter plot of accuracy vs parameter count for model comparison.
    Bubble size proportional to accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, (name, params, acc) in enumerate(zip(model_names, param_counts, accuracies)):
        ax.scatter(
            params, acc * 100,
            s=200 + acc * 500,
            color=colors[i],
            edgecolors="black",
            linewidth=1,
            zorder=5,
            label=name,
        )
        ax.annotate(
            name,
            (params, acc * 100),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_bond_dim_sweep(
    bond_dims: List[int],
    accuracies: List[float],
    param_counts: List[int],
    save_path: Optional[str] = None,
    title: str = "Bond Dimension Sweep",
) -> None:
    """Plot accuracy and param count vs bond dimension."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "steelblue"
    color2 = "coral"

    ax1.plot(bond_dims, [a * 100 for a in accuracies], "o-", color=color1, linewidth=2, markersize=8)
    ax1.set_xlabel("Bond Dimension (χ)", fontsize=13)
    ax1.set_ylabel("Test Accuracy (%)", color=color1, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(bond_dims, param_counts, "s--", color=color2, linewidth=2, markersize=8)
    ax2.set_ylabel("Parameter Count", color=color2, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
