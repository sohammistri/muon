"""Tests for MLP model, data loading, optimizer creation, and training."""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MLP.model import MLP
from MLP.data import get_covertype_loaders, get_year_prediction_loaders, get_mnist_loaders
from MLP.train import create_optimizer, evaluate, OptimizerGroup
from muon import MuonJordan, MuonLLM
from common.metrics import compute_weight_diagnostics, compute_gradient_diagnostics


# ---------------------------------------------------------------------------
# MLP Model tests
# ---------------------------------------------------------------------------

class TestMLPModel:
    def test_default_construction(self):
        model = MLP(input_dim=54, output_dim=7)
        assert isinstance(model.backbone, nn.Sequential)
        assert isinstance(model.head, nn.Linear)
        assert model.head.out_features == 7
        assert model.head.in_features == 128  # last of default [512, 256, 256, 128]

    def test_custom_hidden_dims(self):
        model = MLP(input_dim=100, output_dim=5, hidden_dims=[64, 32])
        assert model.head.out_features == 5
        assert model.head.in_features == 32

    def test_forward_shape_classification(self):
        model = MLP(input_dim=54, output_dim=7, hidden_dims=[32, 16])
        x = torch.randn(4, 54)
        out = model(x)
        assert out.shape == (4, 7)

    def test_forward_shape_regression_squeeze(self):
        """When output_dim=1, forward squeezes the last dim."""
        model = MLP(input_dim=90, output_dim=1, hidden_dims=[32, 16])
        x = torch.randn(4, 90)
        out = model(x)
        assert out.shape == (4,)

    def test_backbone_contains_linear_bn_gelu_dropout(self):
        model = MLP(input_dim=10, output_dim=2, hidden_dims=[16])
        layer_types = [type(m) for m in model.backbone]
        assert nn.Linear in layer_types
        assert nn.BatchNorm1d in layer_types
        assert nn.GELU in layer_types
        assert nn.Dropout in layer_types

    def test_parameter_count_increases_with_hidden_dims(self):
        small = MLP(input_dim=54, output_dim=7, hidden_dims=[32, 16])
        large = MLP(input_dim=54, output_dim=7, hidden_dims=[256, 128, 64])
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestMLPData:
    def test_covertype_metadata(self):
        _, _, input_dim, output_dim = get_covertype_loaders(batch_size=32, seed=0)
        assert input_dim == 54
        assert output_dim == 7

    def test_covertype_batch_shape(self):
        train_loader, _, _, _ = get_covertype_loaders(batch_size=16, seed=0)
        X, y = next(iter(train_loader))
        assert X.shape == (16, 54)
        assert X.dtype == torch.float32
        assert y.shape == (16,)
        assert y.dtype == torch.long

    def test_covertype_label_range(self):
        train_loader, _, _, _ = get_covertype_loaders(batch_size=1024, seed=0)
        X, y = next(iter(train_loader))
        assert y.min() >= 0
        assert y.max() <= 6

    def test_mnist_metadata(self):
        _, _, input_dim, output_dim = get_mnist_loaders(batch_size=32, seed=0)
        assert input_dim == 784
        assert output_dim == 10

    def test_mnist_batch_shape(self):
        train_loader, _, _, _ = get_mnist_loaders(batch_size=16, seed=0)
        X, y = next(iter(train_loader))
        assert X.shape == (16, 784)
        assert y.shape == (16,)
        assert y.dtype == torch.int64

    def test_year_prediction_metadata(self):
        _, _, input_dim, output_dim = get_year_prediction_loaders(batch_size=32, seed=0)
        assert input_dim == 90
        assert output_dim == 1

    def test_year_prediction_batch_shape(self):
        train_loader, _, _, _ = get_year_prediction_loaders(batch_size=16, seed=0)
        X, y = next(iter(train_loader))
        assert X.shape == (16, 90)
        assert X.dtype == torch.float32
        assert y.shape == (16,)
        assert y.dtype == torch.float32

    def test_year_prediction_targets_are_scaled(self):
        train_loader, _, _, _ = get_year_prediction_loaders(batch_size=4096, seed=0)
        all_y = torch.cat([y for _, y in train_loader])
        assert abs(all_y.mean().item()) < 0.1
        assert abs(all_y.std().item() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Optimizer creation tests
# ---------------------------------------------------------------------------

class _Args:
    """Minimal args namespace for create_optimizer."""
    def __init__(self, optim, lr=1e-3, weight_decay=1e-4):
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay


class TestOptimizerCreation:
    @pytest.fixture
    def model(self):
        return MLP(input_dim=54, output_dim=7, hidden_dims=[32, 16])

    def test_sgd(self, model):
        opt = create_optimizer(model, _Args("sgd"))
        assert isinstance(opt, torch.optim.SGD)

    def test_adamw(self, model):
        opt = create_optimizer(model, _Args("adamw"))
        assert isinstance(opt, torch.optim.AdamW)

    def test_muon_jordan(self, model):
        opt = create_optimizer(model, _Args("muon-jordan"))
        assert isinstance(opt, OptimizerGroup)
        assert len(opt.optimizers) == 2
        assert isinstance(opt.optimizers[0], MuonJordan)
        assert isinstance(opt.optimizers[1], torch.optim.AdamW)

    def test_muon_llm(self, model):
        opt = create_optimizer(model, _Args("muon-llm"))
        assert isinstance(opt, OptimizerGroup)
        assert isinstance(opt.optimizers[0], MuonLLM)

    def test_muon_param_split(self, model):
        """Muon should only get 2D backbone Linear weights, not BN, biases, or head."""
        opt = create_optimizer(model, _Args("muon-jordan"))
        muon_opt = opt.optimizers[0]
        adam_opt = opt.optimizers[1]

        muon_param_count = sum(p.numel() for g in muon_opt.param_groups for p in g['params'])
        adam_param_count = sum(p.numel() for g in adam_opt.param_groups for p in g['params'])
        total = sum(p.numel() for p in model.parameters())

        assert muon_param_count + adam_param_count == total
        assert muon_param_count > 0
        assert adam_param_count > 0

        # Verify all muon params are 2D (Linear weights only)
        for g in muon_opt.param_groups:
            for p in g['params']:
                assert p.ndim == 2, "Muon should only receive 2D weight matrices"


# ---------------------------------------------------------------------------
# OptimizerGroup tests
# ---------------------------------------------------------------------------

class TestOptimizerGroup:
    def test_zero_grad_and_step(self):
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(3))
        opt1 = torch.optim.SGD([p1], lr=0.1)
        opt2 = torch.optim.SGD([p2], lr=0.1)
        group = OptimizerGroup(opt1, opt2)

        p1.grad = torch.ones_like(p1)
        p2.grad = torch.ones_like(p2)

        old_p1 = p1.data.clone()
        old_p2 = p2.data.clone()

        group.step()
        assert not torch.equal(p1.data, old_p1)
        assert not torch.equal(p2.data, old_p2)

        group.zero_grad()
        assert p1.grad is None or p1.grad.abs().sum() == 0
        assert p2.grad is None or p2.grad.abs().sum() == 0


# ---------------------------------------------------------------------------
# Training step tests (forward + backward + optimizer step)
# ---------------------------------------------------------------------------

class TestTrainingStep:
    @pytest.fixture
    def classification_setup(self):
        model = MLP(input_dim=54, output_dim=7, hidden_dims=[32, 16])
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(8, 54)
        y = torch.randint(0, 7, (8,))
        return model, criterion, x, y

    @pytest.fixture
    def regression_setup(self):
        model = MLP(input_dim=90, output_dim=1, hidden_dims=[32, 16])
        criterion = nn.MSELoss()
        x = torch.randn(8, 90)
        y = torch.randn(8)
        return model, criterion, x, y

    @pytest.mark.parametrize("optim_name", ["sgd", "adamw", "muon-jordan", "muon-llm"])
    def test_loss_decreases_classification(self, classification_setup, optim_name):
        model, criterion, x, y = classification_setup
        optimizer = create_optimizer(model, _Args(optim_name, lr=1e-2))

        model.train()
        initial_loss = None
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss

    @pytest.mark.parametrize("optim_name", ["sgd", "adamw", "muon-jordan", "muon-llm"])
    def test_loss_decreases_regression(self, regression_setup, optim_name):
        model, criterion, x, y = regression_setup
        optimizer = create_optimizer(model, _Args(optim_name, lr=1e-2))

        model.train()
        initial_loss = None
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss

    def test_gradients_flow_through_all_params(self, classification_setup):
        model, criterion, x, y = classification_setup
        loss = criterion(model(x), y)
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    @pytest.mark.parametrize("optim_name", ["muon-jordan", "muon-llm"])
    def test_muon_preserves_2d_weight_shapes(self, classification_setup, optim_name):
        """After a Muon step, backbone Linear weights should still be 2D."""
        model, criterion, x, y = classification_setup
        optimizer = create_optimizer(model, _Args(optim_name))

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        for name, p in model.named_parameters():
            if 'backbone' in name and 'weight' in name:
                mod = dict(model.named_modules())[name.rsplit('.', 1)[0]]
                if isinstance(mod, nn.Linear):
                    assert p.ndim == 2, f"{name} lost its 2D shape after Muon step"

    def test_weight_decay_applied(self, classification_setup):
        model, criterion, x, y = classification_setup
        optimizer = create_optimizer(model, _Args("muon-jordan", lr=1e-2, weight_decay=0.5))

        # Record initial weight norms
        initial_norms = {}
        for name, p in model.named_parameters():
            if 'weight' in name and p.ndim == 2:
                initial_norms[name] = p.data.norm().item()

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        # At least some weight norms should have decreased due to large weight decay
        decreased = 0
        for name, p in model.named_parameters():
            if name in initial_norms:
                if p.data.norm().item() < initial_norms[name]:
                    decreased += 1
        assert decreased > 0, "Weight decay did not reduce any weight norms"


# ---------------------------------------------------------------------------
# Evaluate function tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_classification_returns_loss_and_accuracy(self):
        model = MLP(input_dim=54, output_dim=7, hidden_dims=[32, 16])
        dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 54), torch.randint(0, 7, (32,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate(model, loader, criterion, "classification", torch.device("cpu"))

        assert "eval/loss" in metrics
        assert "eval/accuracy" in metrics
        assert 0 <= metrics["eval/accuracy"] <= 1
        assert metrics["eval/loss"] > 0

    def test_regression_returns_loss_mse_and_r2(self):
        model = MLP(input_dim=90, output_dim=1, hidden_dims=[32, 16])
        dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 90), torch.randn(32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        criterion = nn.MSELoss()

        metrics = evaluate(model, loader, criterion, "regression", torch.device("cpu"))

        assert "eval/loss" in metrics
        assert "eval/mse" in metrics
        assert "eval/r2" in metrics
        assert metrics["eval/loss"] > 0

    def test_model_back_in_train_mode(self):
        model = MLP(input_dim=54, output_dim=7, hidden_dims=[16])
        model.train()
        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 54), torch.randint(0, 7, (8,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        evaluate(model, loader, nn.CrossEntropyLoss(), "classification", torch.device("cpu"))
        assert model.training

    def test_regression_no_accuracy_key(self):
        model = MLP(input_dim=90, output_dim=1, hidden_dims=[16])
        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 90), torch.randn(8)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        criterion = nn.MSELoss()

        metrics = evaluate(model, loader, criterion, "regression", torch.device("cpu"))
        assert "eval/accuracy" not in metrics


# ---------------------------------------------------------------------------
# Weight diagnostics on MLP
# ---------------------------------------------------------------------------

class TestDiagnosticsOnMLP:
    def test_weight_diagnostics_on_mlp(self):
        model = MLP(input_dim=54, output_dim=7, hidden_dims=[32, 16])
        diag = compute_weight_diagnostics(model)
        keys = list(diag.keys())
        assert any("condition_number" in k for k in keys)
        assert any("spectral_norm" in k for k in keys)

    def test_gradient_diagnostics_after_backward(self):
        model = MLP(input_dim=54, output_dim=7, hidden_dims=[32, 16])
        x = torch.randn(4, 54)
        y = torch.randint(0, 7, (4,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        diag = compute_gradient_diagnostics(model)
        keys = list(diag.keys())
        assert any("grad_norm" in k for k in keys)
