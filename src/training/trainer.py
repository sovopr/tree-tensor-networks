"""
Training loop for TTN experiments.

Features:
- WandB integration for experiment tracking
- Learning rate scheduling (cosine, step)
- Gradient clipping for TTN stability
- Early stopping with patience
- Checkpoint saving/loading
- Comprehensive logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm

from src.utils.metrics import compute_metrics, count_parameters


class Trainer:
    """
    General-purpose trainer for TTN and baseline models.

    Handles the full training loop including validation, early stopping,
    checkpoint management, and optional WandB logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        # Device
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.lr = train_cfg.get("learning_rate", 0.01)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.patience = train_cfg.get("early_stopping_patience", 15)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        opt_name = train_cfg.get("optimizer", "adam")
        wd = train_cfg.get("weight_decay", 1e-5)
        if opt_name == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=wd)
        elif opt_name == "adamw":
            self.optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=wd)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=wd)

        # Scheduler
        sched_name = train_cfg.get("scheduler", "cosine")
        if sched_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
            )
        elif sched_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )
        else:
            self.scheduler = None

        # Logging
        log_cfg = config.get("logging", {})
        self.use_wandb = log_cfg.get("use_wandb", False)
        self.log_interval = log_cfg.get("log_interval", 10)
        self.checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # WandB
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=log_cfg.get("project_name", "ttn-research"),
                    config=config,
                    reinit=True,
                )
            except ImportError:
                print("Warning: wandb not installed, continuing without logging")
                self.use_wandb = False

        # Training state
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        self.best_val_acc = 0.0
        self.patience_counter = 0

        # Model info
        param_info = count_parameters(model)
        print(f"\nModel: {model.__class__.__name__}")
        print(f"  Trainable parameters: {param_info['trainable']:,}")
        print(f"  Total parameters:     {param_info['total']:,}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}, LR: {self.lr}")
        print()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping (important for TTN stability)
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100. * correct / total:.1f}%",
                })

        avg_loss = total_loss / total
        accuracy = correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)

            total_loss += loss.item() * data.size(0)
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(all_targets)
        metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = avg_loss

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Full training loop with validation and early stopping.

        Returns:
            Dictionary with training history and final test results.
        """
        print("=" * 60)
        print(f"Starting training for {self.epochs} epochs")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics["loss"])
            self.train_accs.append(train_metrics["accuracy"])

            # Validate
            val_metrics = self.evaluate(self.val_loader)
            self.val_losses.append(val_metrics["loss"])
            self.val_accs.append(val_metrics["accuracy"])

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Temperature annealing for Adaptive TTN
            if hasattr(self.model, "anneal_temperature"):
                temp = self.model.anneal_temperature()
                if self.use_wandb and self.wandb_run:
                    import wandb
                    wandb.log({"temperature": temp}, step=epoch)

            # Logging
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if self.use_wandb and self.wandb_run:
                import wandb
                wandb.log({
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/f1_macro": val_metrics["f1_macro"],
                    "lr": current_lr,
                    "epoch": epoch,
                }, step=epoch)

            # Best model checkpoint
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        training_time = time.time() - start_time

        # Load best model and evaluate on test set
        self._load_best_checkpoint()
        test_metrics = self.evaluate(self.test_loader)

        print("\n" + "=" * 60)
        print("FINAL TEST RESULTS")
        print("=" * 60)
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"  Precision: {test_metrics['precision_macro']:.4f}")
        print(f"  Recall:    {test_metrics['recall_macro']:.4f}")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Best val accuracy: {self.best_val_acc:.4f}")
        print("=" * 60)

        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log({
                "test/accuracy": test_metrics["accuracy"],
                "test/f1_macro": test_metrics["f1_macro"],
                "test/precision_macro": test_metrics["precision_macro"],
                "test/recall_macro": test_metrics["recall_macro"],
                "training_time_s": training_time,
            })
            wandb.finish()

        # Save results
        results = {
            "test_metrics": test_metrics,
            "best_val_accuracy": self.best_val_acc,
            "training_time_seconds": training_time,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "param_count": count_parameters(self.model),
            "config": self.config,
        }

        results_path = self.checkpoint_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(
                {k: v for k, v in results.items() if not isinstance(v, list) or len(v) < 1000},
                f,
                indent=2,
                default=str,
            )

        return results

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)

    def _load_best_checkpoint(self) -> None:
        """Load the best model checkpoint."""
        path = self.checkpoint_dir / "best_model.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
