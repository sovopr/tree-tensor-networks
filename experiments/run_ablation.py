#!/usr/bin/env python3
"""
Ablation study experiment script.

Runs systematic ablation studies:
- Bond dimension sweep (E5)
- Feature map comparison (E4)
- Topology comparison (E6)
- Disentangler ablation

Usage:
    python experiments/run_ablation.py --config configs/ablation.yaml --ablation bond_dim_sweep
"""

import argparse
import yaml
import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import get_dataloaders
from src.training.trainer import Trainer
from src.utils.metrics import count_parameters
from src.utils.visualization import plot_bond_dim_sweep, plot_parameter_comparison
from experiments.run_classification import build_model


def run_bond_dim_sweep(config: dict, ablation_cfg: dict, output_dir: Path):
    """Sweep bond dimensions and record accuracy vs params."""
    bond_dims = ablation_cfg["bond_dims"]
    seeds = ablation_cfg.get("seeds", [42])
    dataset = ablation_cfg["dataset"]

    results = {"bond_dims": bond_dims, "runs": []}

    for chi in bond_dims:
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"Bond dim χ={chi}, seed={seed}")
            print(f"{'='*50}")

            # Build config for this run
            run_config = {
                "dataset": {"name": dataset, "root": "./data"},
                "feature_map": {"type": ablation_cfg.get("feature_map", "trigonometric"), "local_dim": 2},
                "model": {
                    "type": ablation_cfg.get("model_type", "ttn"),
                    "bond_dim": chi,
                    "num_classes": 10,
                    "init_method": "qr",
                },
                "training": config.get("training", {}),
                "logging": {"use_wandb": False, "checkpoint_dir": str(output_dir / f"chi{chi}_seed{seed}")},
                "seed": seed,
            }

            torch.manual_seed(seed)
            train_loader, val_loader, test_loader, info = get_dataloaders(
                name=dataset, batch_size=256, seed=seed,
            )

            model = build_model(run_config, input_dim=info["input_dim"])
            trainer = Trainer(model, train_loader, val_loader, test_loader, run_config)
            run_results = trainer.train()

            results["runs"].append({
                "bond_dim": chi,
                "seed": seed,
                "test_accuracy": run_results["test_metrics"]["accuracy"],
                "test_f1": run_results["test_metrics"]["f1_macro"],
                "param_count": run_results["param_count"]["trainable"],
                "training_time": run_results["training_time_seconds"],
            })

    # Average over seeds
    avg_results = {}
    for chi in bond_dims:
        chi_runs = [r for r in results["runs"] if r["bond_dim"] == chi]
        avg_results[chi] = {
            "accuracy_mean": sum(r["test_accuracy"] for r in chi_runs) / len(chi_runs),
            "accuracy_std": (sum((r["test_accuracy"] - sum(r2["test_accuracy"] for r2 in chi_runs) / len(chi_runs))**2 for r in chi_runs) / len(chi_runs)) ** 0.5,
            "param_count": chi_runs[0]["param_count"],
        }

    results["averaged"] = avg_results

    # Plot
    accs = [avg_results[chi]["accuracy_mean"] for chi in bond_dims]
    params = [avg_results[chi]["param_count"] for chi in bond_dims]
    plot_bond_dim_sweep(bond_dims, accs, params, save_path=str(output_dir / "bond_dim_sweep.png"))

    return results


def run_feature_map_sweep(config: dict, ablation_cfg: dict, output_dir: Path):
    """Compare different feature maps."""
    feature_maps = ablation_cfg["feature_maps"]
    seeds = ablation_cfg.get("seeds", [42])
    dataset = ablation_cfg["dataset"]
    bond_dim = ablation_cfg["bond_dim"]

    results = {"feature_maps": [], "runs": []}

    for fmap_cfg in feature_maps:
        fmap_type = fmap_cfg["type"]
        results["feature_maps"].append(fmap_type)

        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"Feature map: {fmap_type}, seed={seed}")
            print(f"{'='*50}")

            run_config = {
                "dataset": {"name": dataset, "root": "./data"},
                "feature_map": fmap_cfg,
                "model": {"type": "ttn", "bond_dim": bond_dim, "num_classes": 10, "init_method": "qr"},
                "training": config.get("training", {}),
                "logging": {"use_wandb": False, "checkpoint_dir": str(output_dir / f"fmap_{fmap_type}_seed{seed}")},
                "seed": seed,
            }

            torch.manual_seed(seed)
            train_loader, val_loader, test_loader, info = get_dataloaders(
                name=dataset, batch_size=256, seed=seed,
            )

            model = build_model(run_config, input_dim=info["input_dim"])
            trainer = Trainer(model, train_loader, val_loader, test_loader, run_config)
            run_results = trainer.train()

            results["runs"].append({
                "feature_map": fmap_type,
                "seed": seed,
                "test_accuracy": run_results["test_metrics"]["accuracy"],
                "param_count": run_results["param_count"]["trainable"],
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="TTN Ablation Studies")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["bond_dim_sweep", "feature_map_sweep", "topology_sweep", "disentangler_sweep"])
    parser.add_argument("--output_dir", type=str, default="./results/ablation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir) / args.ablation
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_cfg = config["ablations"][args.ablation]

    if args.ablation == "bond_dim_sweep":
        results = run_bond_dim_sweep(config, ablation_cfg, output_dir)
    elif args.ablation == "feature_map_sweep":
        results = run_feature_map_sweep(config, ablation_cfg, output_dir)
    else:
        print(f"Ablation '{args.ablation}' implementation in progress")
        results = {}

    # Save results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Ablation results saved to {output_dir}")


if __name__ == "__main__":
    main()
