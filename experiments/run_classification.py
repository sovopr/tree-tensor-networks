#!/usr/bin/env python3
"""
Main classification experiment script.

Runs TTN and baseline models on MNIST/Fashion-MNIST/CIFAR-10.
Corresponds to Experiments E1, E2, E3 in the experiment matrix.

Usage:
    python experiments/run_classification.py --config configs/mnist.yaml
    python experiments/run_classification.py --config configs/mnist.yaml --debug --max_samples 1000
    python experiments/run_classification.py --config configs/mnist.yaml --model_type adaptive_ttn
"""

import argparse
import yaml
import torch
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import get_dataloaders
from src.models.ttn import TreeTensorNetwork
from src.models.augmented_ttn import AugmentedTTN
from src.models.adaptive_ttn import AdaptiveTTN
from src.models.baselines import LogisticRegressionModel, MLPModel, LightweightCNN, MPSClassifier
from src.training.trainer import Trainer
from src.utils.metrics import count_parameters
from src.utils.visualization import plot_training_curves


def build_model(config: dict, model_type: str = None, input_dim: int = 784) -> torch.nn.Module:
    """Build model from config."""
    model_cfg = config["model"]
    mtype = model_type or model_cfg.get("type", "ttn")
    num_classes = model_cfg.get("num_classes", 10)
    bond_dim = model_cfg.get("bond_dim", 8)
    fmap_config = config.get("feature_map", {"type": "trigonometric", "local_dim": 2})

    if mtype == "ttn":
        return TreeTensorNetwork(
            input_dim=input_dim,
            num_classes=num_classes,
            bond_dim=bond_dim,
            feature_map_config=fmap_config,
            init_method=model_cfg.get("init_method", "qr"),
        )
    elif mtype == "augmented_ttn":
        return AugmentedTTN(
            input_dim=input_dim,
            num_classes=num_classes,
            bond_dim=bond_dim,
            feature_map_config=fmap_config,
            init_method=model_cfg.get("init_method", "qr"),
            use_disentanglers=model_cfg.get("use_disentanglers", True),
        )
    elif mtype == "adaptive_ttn":
        return AdaptiveTTN(
            input_dim=input_dim,
            num_classes=num_classes,
            bond_dim=bond_dim,
            feature_map_config=fmap_config,
            init_method=model_cfg.get("init_method", "qr"),
            initial_temperature=model_cfg.get("gumbel_temperature", 1.0),
            anneal_rate=model_cfg.get("gumbel_anneal_rate", 0.003),
        )
    elif mtype == "logistic_regression":
        return LogisticRegressionModel(input_dim=input_dim, num_classes=num_classes)
    elif mtype == "mlp":
        return MLPModel(input_dim=input_dim, num_classes=num_classes, hidden_dim=128)
    elif mtype == "cnn":
        dataset_name = config["dataset"]["name"]
        channels = 1 if dataset_name in ["mnist", "fashion_mnist"] else 3
        img_size = 28 if dataset_name in ["mnist", "fashion_mnist"] else 32
        return LightweightCNN(input_channels=channels, num_classes=num_classes, input_size=img_size)
    elif mtype == "mps":
        return MPSClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            bond_dim=bond_dim,
            feature_map_config=fmap_config,
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def main():
    parser = argparse.ArgumentParser(description="TTN Classification Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_type", type=str, default=None, help="Override model type")
    parser.add_argument("--bond_dim", type=int, default=None, help="Override bond dimension")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Overrides
    if args.model_type:
        config["model"]["type"] = args.model_type
    if args.bond_dim:
        config["model"]["bond_dim"] = args.bond_dim
    if args.seed:
        config["seed"] = args.seed
    if args.debug:
        config["training"]["epochs"] = 5
        config["logging"]["use_wandb"] = False

    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load data
    dataset_cfg = config["dataset"]
    train_loader, val_loader, test_loader, data_info = get_dataloaders(
        name=dataset_cfg["name"],
        root=dataset_cfg.get("root", "./data"),
        batch_size=config["training"]["batch_size"],
        num_workers=config.get("num_workers", 4),
        max_samples=args.max_samples,
        seed=seed,
    )

    input_dim = data_info["input_dim"]
    print(f"\nDataset: {dataset_cfg['name']}")
    print(f"  Input dim: {input_dim}")
    print(f"  Train samples: {data_info['num_train']}")
    print(f"  Val samples: {data_info['num_val']}")
    print(f"  Test samples: {data_info['num_test']}")

    # Build model
    model_type = config["model"].get("type", "ttn")
    model = build_model(config, model_type, input_dim)

    # MI initialization for adaptive TTN
    if model_type == "adaptive_ttn" and hasattr(model, "initialize_from_mi"):
        print("\nInitializing adaptive TTN from mutual information...")
        # Get a batch of training data for MI computation
        all_data = []
        for data, _ in train_loader:
            all_data.append(data)
            if len(all_data) * data.shape[0] >= 5000:
                break
        all_data = torch.cat(all_data)[:5000]
        model.initialize_from_mi(all_data)

    # Setup output directory
    output_dir = Path(args.output_dir) / dataset_cfg["name"] / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    config["logging"]["checkpoint_dir"] = str(output_dir / "checkpoints")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=config.get("device", "auto"),
    )

    results = trainer.train()

    # Save training curves
    plot_training_curves(
        trainer.train_losses,
        trainer.val_losses,
        trainer.train_accs,
        trainer.val_accs,
        save_path=str(output_dir / "training_curves.png"),
        title=f"{model_type} on {dataset_cfg['name']}",
    )

    # Save results summary
    summary = {
        "model_type": model_type,
        "dataset": dataset_cfg["name"],
        "bond_dim": config["model"].get("bond_dim"),
        "feature_map": config.get("feature_map", {}).get("type"),
        "test_accuracy": results["test_metrics"]["accuracy"],
        "test_f1": results["test_metrics"]["f1_macro"],
        "param_count": results["param_count"],
        "training_time": results["training_time_seconds"],
        "best_val_acc": results["best_val_accuracy"],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")

    return results


if __name__ == "__main__":
    main()
