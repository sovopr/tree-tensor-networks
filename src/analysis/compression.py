"""
Compression analysis utilities.

Compares parameter efficiency and computational cost across models,
and analyzes scaling behavior as input dimensionality grows.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Any, Optional

from src.utils.metrics import count_parameters, measure_inference_time, measure_memory_usage


def compression_analysis(
    models: Dict[str, nn.Module],
    input_shape: tuple,
    device: str = "cpu",
    reference_model: str = "mlp",
) -> Dict[str, Dict[str, Any]]:
    """
    Comprehensive compression analysis across multiple models.

    Args:
        models: Dict mapping model name → model
        input_shape: Input tensor shape (without batch)
        device: Device for timing measurements
        reference_model: Model to use as compression reference

    Returns:
        Dict mapping model name → analysis results
    """
    results = {}
    ref_params = None

    for name, model in models.items():
        model = model.to(device)
        params = count_parameters(model)
        timing = measure_inference_time(model, input_shape, device)
        memory = measure_memory_usage(model, input_shape, device)

        if name == reference_model:
            ref_params = params["trainable"]

        results[name] = {
            "params": params,
            "timing": timing,
            "memory": memory,
        }

    # Add compression ratios relative to reference
    if ref_params is not None:
        for name in results:
            model_params = results[name]["params"]["trainable"]
            results[name]["compression_ratio"] = ref_params / max(model_params, 1)

    return results


def scaling_analysis(
    model_class: type,
    input_dims: List[int] = [64, 128, 256, 512, 1024, 2048],
    bond_dims: List[int] = [4, 8, 16, 32],
    num_classes: int = 10,
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Analyze how TTN parameter count and FLOPs scale with input dimension.

    For a TTN with input dim N and bond dim χ:
    - Parameters: O(N × d² × χ) for bottom layer + O(N/2 × χ² × χ) for rest
    - Total: O(N × χ × max(d², χ²))
    - Compared to MLP: O(N × H + H²) where H is hidden dim

    This shows that TTN scales linearly with N (like MLP) but with
    much smaller constants when χ is small.

    Args:
        model_class: TTN model class to analyze
        input_dims: List of input dimensions to test
        bond_dims: List of bond dimensions to test
        num_classes: Number of output classes
        **model_kwargs: Additional model arguments

    Returns:
        Dict with scaling data
    """
    results = {
        "input_dims": input_dims,
        "bond_dims": bond_dims,
        "param_counts": {},  # (bond_dim, input_dim) → count
        "theoretical_scaling": {},
    }

    for chi in bond_dims:
        counts = []
        for N in input_dims:
            try:
                model = model_class(
                    input_dim=N,
                    num_classes=num_classes,
                    bond_dim=chi,
                    **model_kwargs,
                )
                params = count_parameters(model)
                counts.append(params["trainable"])
            except Exception as e:
                counts.append(None)

        results["param_counts"][chi] = counts

        # Theoretical: O(N * d^2 * chi) for first layer + O(N * chi^3) for rest
        # Simplified: roughly O(N * chi * max(d^2, chi^2))
        d = 2  # local dim
        theoretical = [
            N * d * d * chi + (N - 1) * chi * chi * chi  # approximate
            for N in input_dims
        ]
        results["theoretical_scaling"][chi] = theoretical

    # Also compute MLP scaling for comparison
    mlp_counts = []
    for N in input_dims:
        # MLP with hidden_dim chosen to roughly match TTN(chi=8) params
        hidden = 128
        mlp_params = N * hidden + hidden * (hidden // 2) + (hidden // 2) * num_classes
        mlp_counts.append(mlp_params)
    results["mlp_scaling"] = mlp_counts

    return results
