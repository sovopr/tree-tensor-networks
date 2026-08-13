#!/usr/bin/env python3
"""
Interpretability experiment — Entanglement entropy analysis.

Runs entanglement analysis on trained TTN models:
1. Bond entanglement at each tree level
2. Feature importance ranking via entanglement
3. Class-conditional entanglement maps
4. Layer-wise entanglement profiles

Corresponds to Experiment E8 in the experiment matrix.

Usage:
    python experiments/run_interpretability.py --config configs/mnist.yaml --checkpoint results/mnist/ttn/checkpoints/best_model.pt
"""

import argparse
import yaml
import torch
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import get_dataloaders
from src.models.ttn import TreeTensorNetwork
from src.models.augmented_ttn import AugmentedTTN
from src.analysis.entanglement import (
    compute_bond_entanglement,
    compute_feature_entanglement_map,
    layer_wise_entanglement_profile,
)
from src.analysis.interpretability import (
    feature_importance_from_entanglement,
    class_conditional_entanglement,
)
from src.utils.visualization import (
    plot_entanglement_map,
    plot_ttn_structure,
)
from experiments.run_classification import build_model


def main():
    parser = argparse.ArgumentParser(description="TTN Interpretability Analysis")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results/interpretability")
    parser.add_argument("--num_samples", type=int, default=500, help="Samples for analysis")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data
    _, _, test_loader, data_info = get_dataloaders(
        name=config["dataset"]["name"],
        root=config["dataset"].get("root", "./data"),
        batch_size=256,
        num_workers=2,
    )

    # Build and load model
    model = build_model(config, input_dim=data_info["input_dim"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect test data
    all_data, all_labels = [], []
    for data, labels in test_loader:
        all_data.append(data)
        all_labels.append(labels)
    all_data = torch.cat(all_data)[:args.num_samples]
    all_labels = torch.cat(all_labels)[:args.num_samples]

    # 1. Bond entanglement
    print("\n1. Computing bond entanglement...")
    bond_ent = compute_bond_entanglement(model)
    print(f"   Layers: {len(bond_ent)}")
    for i, layer_ent in enumerate(bond_ent):
        if layer_ent:
            print(f"   Layer {i}: mean={np.mean(layer_ent):.4f}, max={np.max(layer_ent):.4f}, nodes={len(layer_ent)}")

    # 2. Layer-wise profile
    print("\n2. Layer-wise entanglement profile:")
    profile = layer_wise_entanglement_profile(model)
    for i, (layer, mean_ent) in enumerate(zip(profile["layer"], profile["mean_entropy"])):
        print(f"   Layer {layer}: mean entropy = {mean_ent:.4f}")

    # 3. Feature entanglement map
    print("\n3. Computing feature entanglement map...")
    ent_map = compute_feature_entanglement_map(model, all_data, args.num_samples)
    plot_entanglement_map(
        ent_map,
        save_path=str(output_dir / "feature_entanglement_map.png"),
        title=f"Feature Entanglement — {config['dataset']['name']}",
    )
    print(f"   Saved to {output_dir / 'feature_entanglement_map.png'}")

    # 4. Feature importance
    print("\n4. Feature importance ranking:")
    importance = feature_importance_from_entanglement(model, all_data, args.num_samples)
    top_10 = importance["ranking"][:10]
    print(f"   Top 10 most important features: {top_10}")
    print(f"   Their scores: {importance['importance_scores'][top_10]}")

    # Save importance as image (reshape to original image dims)
    dataset_name = config["dataset"]["name"]
    if dataset_name in ["mnist", "fashion_mnist"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = importance["importance_scores"][:784]
        img = scores.reshape(28, 28)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="hot", interpolation="nearest")
        ax.set_title("Feature Importance (Entanglement-based)", fontsize=13)
        ax.axis("off")
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
        plt.savefig(str(output_dir / "feature_importance_heatmap.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   Saved importance heatmap to {output_dir / 'feature_importance_heatmap.png'}")

    # 5. Class-conditional entanglement
    print("\n5. Class-conditional entanglement maps:")
    class_maps = class_conditional_entanglement(
        model, all_data, all_labels,
        num_classes=config["model"].get("num_classes", 10),
        num_samples_per_class=50,
    )
    for c, cmap in class_maps.items():
        plot_entanglement_map(
            cmap,
            save_path=str(output_dir / f"class_{c}_entanglement.png"),
            title=f"Class {c} Entanglement",
        )
    print(f"   Saved {len(class_maps)} class-conditional maps")

    # 6. TTN structure visualization
    print("\n6. Visualizing TTN structure...")
    if hasattr(model, "get_tree_info"):
        info = model.get_tree_info()
        plot_ttn_structure(
            num_features=min(info["input_dim"], 64),  # limit for readability
            tree_layers=info["tree_structure"],
            entanglement_values=bond_ent,
            save_path=str(output_dir / "ttn_structure.png"),
            title="TTN Structure with Entanglement Coloring",
        )

    # Save all results
    results = {
        "bond_entanglement": bond_ent,
        "layer_profile": profile,
        "top_features": top_10.tolist(),
        "importance_scores": importance["importance_scores"].tolist(),
    }
    with open(output_dir / "interpretability_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ All results saved to {output_dir}")


if __name__ == "__main__":
    main()
