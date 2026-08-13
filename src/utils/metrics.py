"""
Evaluation metrics for TTN experiments.

Provides comprehensive metrics including classification performance,
parameter efficiency, computational cost, and memory usage.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> str:
    """Get detailed classification report as string."""
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Get confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.

    Returns:
        Dict with 'trainable', 'total', and 'non_trainable' counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "non_trainable": total - trainable,
    }


def estimate_flops(model: nn.Module, input_shape: tuple, device: str = "cpu") -> int:
    """
    Estimate FLOPs for a forward pass.

    Uses a simple hook-based counter. For TTN models, the main cost is
    in tensor contractions (einsum operations).

    Args:
        model: The model
        input_shape: Shape of a single input (without batch dim)
        device: Device to run on

    Returns:
        Estimated FLOPs count
    """
    flop_count = [0]

    def _count_linear(module, inp, out):
        # Linear: 2 * in_features * out_features (multiply-add)
        if isinstance(module, nn.Linear):
            flop_count[0] += 2 * module.in_features * module.out_features

    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(_count_linear))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, *input_shape).to(device)
        try:
            model(dummy)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    return flop_count[0]


def measure_inference_time(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cpu",
    num_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Measure inference latency.

    Args:
        model: The model
        input_shape: Input shape (without batch dim)
        device: Device
        num_runs: Number of timing runs
        warmup: Number of warmup runs

    Returns:
        Dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms'
    """
    model.eval()
    dummy = torch.randn(1, *input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy)
            if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "median_ms": float(np.median(times)),
    }


def measure_memory_usage(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure peak GPU memory usage during forward pass.

    Args:
        model: The model
        input_shape: Input shape
        device: Must be 'cuda'

    Returns:
        Dict with memory stats in MB
    """
    if not torch.cuda.is_available() or device == "cpu":
        # Return parameter memory estimate for CPU
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        return {
            "param_memory_mb": param_mem / (1024 ** 2),
            "peak_memory_mb": -1,  # Cannot measure on CPU
        }

    model = model.to(device)
    dummy = torch.randn(1, *input_shape).to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    with torch.no_grad():
        model(dummy)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    return {
        "param_memory_mb": param_mem,
        "peak_memory_mb": peak_mem,
    }
