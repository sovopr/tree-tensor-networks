#!/usr/bin/env python3
"""
Tests for core TTN operations and models.
Run with: pytest tests/ -v
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.tensor_ops import (
    qr_init,
    random_init,
    contract_pair,
    contract_pair_batched,
    build_binary_tree_structure,
)
from src.data.feature_maps import (
    TrigonometricFeatureMap,
    FourierFeatureMap,
    POVMFeatureMap,
    get_feature_map,
)
from src.models.ttn import TreeTensorNetwork
from src.models.augmented_ttn import AugmentedTTN
from src.models.adaptive_ttn import AdaptiveTTN
from src.models.baselines import LogisticRegressionModel, MLPModel, LightweightCNN, MPSClassifier
from src.models.born_machine import TTNBornMachine
from src.models.tensorized_attn import TensorizedLinear, TensorizedAttention
from src.analysis.entanglement import compute_bond_entropy, compute_bond_entanglement
from src.utils.metrics import count_parameters


# ============================================================
# Tensor Operations Tests
# ============================================================

class TestTensorOps:
    def test_qr_init_shape(self):
        tensor = qr_init((2, 2, 8))
        assert tensor.shape == (2, 2, 8)

    def test_qr_init_approximate_isometry(self):
        """QR-initialized tensor should be approximately isometric."""
        tensor = qr_init((4, 4, 8))
        mat = tensor.reshape(16, 8)
        gram = mat.T @ mat
        identity = torch.eye(8)
        assert torch.allclose(gram, identity, atol=1e-5)

    def test_contract_pair(self):
        batch = 32
        d_left, d_right, chi = 2, 2, 8
        left = torch.randn(batch, d_left)
        right = torch.randn(batch, d_right)
        node = torch.randn(d_left, d_right, chi)
        output = contract_pair(left, right, node)
        assert output.shape == (batch, chi)

    def test_contract_pair_batched_matches_einsum(self):
        """Batched contraction should give same result as einsum version."""
        batch = 64
        d_left, d_right, chi = 2, 2, 8
        left = torch.randn(batch, d_left)
        right = torch.randn(batch, d_right)
        node = torch.randn(d_left, d_right, chi)

        out1 = contract_pair(left, right, node)
        out2 = contract_pair_batched(left, right, node)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_build_binary_tree(self):
        layers, num_padded = build_binary_tree_structure(8)
        assert num_padded == 8
        assert len(layers) == 3  # log2(8) = 3
        assert len(layers[0]) == 4  # 4 pairs at bottom
        assert len(layers[1]) == 2
        assert len(layers[2]) == 1

    def test_build_binary_tree_non_power_of_2(self):
        layers, num_padded = build_binary_tree_structure(6)
        assert num_padded == 8  # padded to next power of 2
        assert len(layers) == 3


# ============================================================
# Feature Map Tests
# ============================================================

class TestFeatureMaps:
    def test_trig_map_shape(self):
        fmap = TrigonometricFeatureMap(local_dim=2)
        x = torch.rand(32, 784)
        out = fmap(x)
        assert out.shape == (32, 784, 2)

    def test_trig_map_normalization(self):
        """Trig feature map should produce unit-norm vectors."""
        fmap = TrigonometricFeatureMap()
        x = torch.rand(10, 100)
        out = fmap(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_fourier_map_shape(self):
        fmap = FourierFeatureMap(local_dim=8, num_frequencies=4)
        x = torch.rand(32, 784)
        out = fmap(x)
        assert out.shape == (32, 784, 8)

    def test_fourier_map_learnable(self):
        fmap = FourierFeatureMap(learnable_frequencies=True)
        assert any(p.requires_grad for p in fmap.parameters())

    def test_povm_map_shape(self):
        fmap = POVMFeatureMap(povm_dim=4)
        x = torch.rand(32, 784)
        out = fmap(x)
        assert out.shape == (32, 784, 4)

    def test_povm_map_positive(self):
        """POVM output should be non-negative (softmax)."""
        fmap = POVMFeatureMap(povm_dim=4)
        x = torch.rand(10, 50)
        out = fmap(x)
        assert (out >= 0).all()

    def test_factory(self):
        fmap = get_feature_map({"type": "trigonometric", "local_dim": 2})
        assert isinstance(fmap, TrigonometricFeatureMap)


# ============================================================
# Model Tests
# ============================================================

class TestTTN:
    def test_forward_shape(self):
        model = TreeTensorNetwork(input_dim=64, num_classes=10, bond_dim=4)
        x = torch.rand(16, 64)
        out = model(x)
        assert out.shape == (16, 10)

    def test_forward_mnist_dim(self):
        model = TreeTensorNetwork(input_dim=784, num_classes=10, bond_dim=8)
        x = torch.rand(8, 784)
        out = model(x)
        assert out.shape == (8, 10)

    def test_gradient_flow(self):
        """Ensure gradients flow through the entire TTN."""
        model = TreeTensorNetwork(input_dim=16, num_classes=10, bond_dim=4)
        x = torch.rand(4, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_intermediate_states(self):
        model = TreeTensorNetwork(input_dim=16, num_classes=10, bond_dim=4)
        x = torch.rand(4, 16)
        states = model.get_intermediate_states(x)
        assert len(states) > 1  # at least input + one layer


class TestAugmentedTTN:
    def test_forward_shape(self):
        model = AugmentedTTN(input_dim=64, num_classes=10, bond_dim=4)
        x = torch.rand(16, 64)
        out = model(x)
        assert out.shape == (16, 10)

    def test_disentanglers_add_params(self):
        """Augmented TTN should have more parameters than standard TTN."""
        ttn = TreeTensorNetwork(input_dim=64, num_classes=10, bond_dim=4)
        attn = AugmentedTTN(input_dim=64, num_classes=10, bond_dim=4)
        ttn_params = count_parameters(ttn)["trainable"]
        attn_params = count_parameters(attn)["trainable"]
        assert attn_params > ttn_params


class TestAdaptiveTTN:
    def test_forward_shape(self):
        model = AdaptiveTTN(input_dim=16, num_classes=10, bond_dim=4)
        x = torch.rand(8, 16)
        out = model(x)
        assert out.shape == (8, 10)

    def test_temperature_annealing(self):
        model = AdaptiveTTN(input_dim=16, bond_dim=4, initial_temperature=1.0)
        initial_temp = model.temperature
        model.anneal_temperature()
        assert model.temperature < initial_temp


class TestBaselines:
    def test_logreg(self):
        model = LogisticRegressionModel(784, 10)
        x = torch.rand(16, 784)
        assert model(x).shape == (16, 10)

    def test_mlp(self):
        model = MLPModel(784, 128, 10)
        x = torch.rand(16, 784)
        assert model(x).shape == (16, 10)

    def test_cnn(self):
        model = LightweightCNN(1, 10, 28)
        x = torch.rand(16, 784)
        assert model(x).shape == (16, 10)

    def test_mps(self):
        model = MPSClassifier(input_dim=64, num_classes=10, bond_dim=4)
        x = torch.rand(8, 64)
        assert model(x).shape == (8, 10)


class TestBornMachine:
    def test_amplitude(self):
        model = TTNBornMachine(input_dim=16, bond_dim=4)
        x = torch.rand(8, 16)
        amp = model.amplitude(x)
        assert amp.shape == (8,)

    def test_log_probability(self):
        model = TTNBornMachine(input_dim=16, bond_dim=4)
        x = torch.rand(8, 16)
        log_p = model.log_probability(x)
        assert log_p.shape == (8,)

    def test_nll_loss(self):
        model = TTNBornMachine(input_dim=16, bond_dim=4)
        x = torch.rand(8, 16)
        loss = model.nll_loss(x)
        assert loss.dim() == 0  # scalar


class TestTensorizedAttn:
    def test_tensorized_linear(self):
        layer = TensorizedLinear(64, 64, tt_rank=4)
        x = torch.rand(8, 64)
        out = layer(x)
        assert out.shape == (8, 64)

    def test_compression_ratio(self):
        layer = TensorizedLinear(256, 256, tt_rank=4)
        ratio = layer.compression_ratio()
        assert ratio > 1.0  # should be compressed

    def test_tensorized_attention(self):
        attn = TensorizedAttention(d_model=64, num_heads=4, tt_rank=4)
        x = torch.rand(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)


# ============================================================
# Entanglement Analysis Tests
# ============================================================

class TestEntanglement:
    def test_bond_entropy_range(self):
        """Entropy should be non-negative."""
        tensor = torch.randn(2, 2, 8)
        entropy = compute_bond_entropy(tensor)
        assert entropy >= 0

    def test_identity_entropy_zero(self):
        """A rank-1 tensor should have zero entropy."""
        # Rank-1: outer product of two vectors
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([1.0, 0.0])
        tensor = torch.einsum("i,j->ij", a, b).unsqueeze(-1)  # (2, 2, 1)
        entropy = compute_bond_entropy(tensor)
        assert entropy < 0.01  # approximately zero

    def test_bond_entanglement_layers(self):
        model = TreeTensorNetwork(input_dim=16, num_classes=10, bond_dim=4)
        entropies = compute_bond_entanglement(model)
        assert len(entropies) > 0
        for layer_ent in entropies:
            for e in layer_ent:
                assert e >= 0


# ============================================================
# Dynamic Bond TTN Tests
# ============================================================

class TestDynamicBondTTN:
    def test_forward_shape(self):
        from src.models.dynamic_bond_ttn import DynamicBondTTN
        model = DynamicBondTTN(input_dim=16, num_classes=10, candidate_dims=[2, 4, 8])
        x = torch.rand(4, 16)
        out = model(x)
        assert out.shape == (4, 10)

    def test_selected_dimensions(self):
        from src.models.dynamic_bond_ttn import DynamicBondTTN
        model = DynamicBondTTN(input_dim=16, num_classes=10, candidate_dims=[2, 4, 8])
        dims = model.get_selected_dimensions()
        assert len(dims) == model.num_tree_layers
        for d in dims:
            assert d in [2, 4, 8]

    def test_complexity_penalty(self):
        from src.models.dynamic_bond_ttn import DynamicBondTTN
        model = DynamicBondTTN(input_dim=16, num_classes=10, candidate_dims=[2, 4, 8])
        penalty = model.get_complexity_penalty()
        assert penalty.dim() == 0  # scalar
        assert 0.0 <= penalty.item() <= 1.0

    def test_temperature_annealing(self):
        from src.models.dynamic_bond_ttn import DynamicBondTTN
        model = DynamicBondTTN(input_dim=16, num_classes=10, candidate_dims=[2, 4, 8])
        t0 = model.temperature
        t1 = model.anneal_temperature()
        assert t1 < t0

    def test_gradient_flow(self):
        from src.models.dynamic_bond_ttn import DynamicBondTTN
        model = DynamicBondTTN(input_dim=16, num_classes=10, candidate_dims=[2, 4])
        model.train()
        x = torch.rand(4, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that bond selector logits have gradients
        for layer in model.ttn_layers:
            assert layer.bond_selector.selection_logits.grad is not None


class TestFullyAdaptiveTTN:
    def test_forward_shape(self):
        from src.models.dynamic_bond_ttn import FullyAdaptiveTTN
        model = FullyAdaptiveTTN(input_dim=16, num_classes=10, candidate_dims=[2, 4, 8])
        x = torch.rand(4, 16)
        out = model(x)
        assert out.shape == (4, 10)

    def test_gradient_flow(self):
        from src.models.dynamic_bond_ttn import FullyAdaptiveTTN
        model = FullyAdaptiveTTN(input_dim=16, num_classes=10, candidate_dims=[2, 4])
        model.train()
        x = torch.rand(4, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

