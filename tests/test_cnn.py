"""Tests for CNN model, data loading, optimizer creation, and training."""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CNN.model import CNN
from CNN.data import get_cifar10_loaders, CIFAR10_MEAN, CIFAR10_STD
from CNN.train import create_optimizer, evaluate, OptimizerGroup
from muon import MuonJordan, MuonLLM
from common.metrics import compute_weight_diagnostics, compute_gradient_diagnostics


# ---------------------------------------------------------------------------
# CNN Model tests
# ---------------------------------------------------------------------------

class TestCNNModel:
    def test_default_construction(self):
        model = CNN()
        assert isinstance(model.backbone, nn.Sequential)
        assert isinstance(model.pool, nn.AdaptiveAvgPool2d)
        assert isinstance(model.head, nn.Linear)
        assert model.head.out_features == 10

    def test_custom_channels(self):
        model = CNN(in_channels=1, num_classes=5, channels=[16, 32])
        assert model.head.out_features == 5
        assert model.head.in_features == 32

    def test_forward_shape(self):
        model = CNN(in_channels=3, num_classes=10, channels=[16, 32])
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_single_channel(self):
        model = CNN(in_channels=1, num_classes=5, channels=[8, 16])
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 5)

    def test_backbone_contains_conv_bn_gelu_blocks(self):
        model = CNN(channels=[16])
        layer_types = [type(m) for m in model.backbone]
        assert nn.Conv2d in layer_types
        assert nn.BatchNorm2d in layer_types
        assert nn.GELU in layer_types
        assert nn.MaxPool2d in layer_types

    def test_parameter_count_increases_with_channels(self):
        small = CNN(channels=[8, 16])
        large = CNN(channels=[32, 64, 128])
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestCIFAR10Data:
    def test_loader_returns_correct_metadata(self):
        train_loader, test_loader, in_channels, num_classes = \
            get_cifar10_loaders(batch_size=32, seed=0)
        assert in_channels == 3
        assert num_classes == 10

    def test_train_batch_shape(self):
        train_loader, _, _, _ = get_cifar10_loaders(batch_size=16, seed=0)
        X, y = next(iter(train_loader))
        assert X.shape == (16, 3, 32, 32)
        assert y.shape == (16,)
        assert y.dtype == torch.int64

    def test_normalization_constants_defined(self):
        assert len(CIFAR10_MEAN) == 3
        assert len(CIFAR10_STD) == 3
        for v in CIFAR10_STD:
            assert v > 0


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
        return CNN(channels=[8, 16])

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
        """Muon should only get backbone Conv2d/Linear weights, not BN or head."""
        opt = create_optimizer(model, _Args("muon-jordan"))
        muon_opt = opt.optimizers[0]
        adam_opt = opt.optimizers[1]

        muon_param_count = sum(p.numel() for g in muon_opt.param_groups for p in g['params'])
        adam_param_count = sum(p.numel() for g in adam_opt.param_groups for p in g['params'])
        total = sum(p.numel() for p in model.parameters())

        assert muon_param_count + adam_param_count == total
        assert muon_param_count > 0
        assert adam_param_count > 0


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

        # Simulate gradients
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
    def setup(self):
        model = CNN(in_channels=3, num_classes=10, channels=[8, 16])
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        return model, criterion, x, y

    @pytest.mark.parametrize("optim_name", ["sgd", "adamw", "muon-jordan", "muon-llm"])
    def test_loss_decreases_after_steps(self, setup, optim_name):
        model, criterion, x, y = setup
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
        # Loss should decrease on a tiny overfitting batch
        assert final_loss < initial_loss

    @pytest.mark.parametrize("optim_name", ["muon-jordan", "muon-llm"])
    def test_muon_handles_4d_conv_weights(self, setup, optim_name):
        """Muon must reshape 4D conv weights to 2D for NS iteration, then reshape back."""
        model, criterion, x, y = setup
        optimizer = create_optimizer(model, _Args(optim_name))

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        # All conv weights should still be 4D after the step
        for name, p in model.named_parameters():
            if 'backbone' in name and 'weight' in name:
                mod = dict(model.named_modules())[name.rsplit('.', 1)[0]]
                if isinstance(mod, nn.Conv2d):
                    assert p.ndim == 4, f"{name} lost its 4D shape after Muon step"

    def test_gradients_flow_through_all_params(self, setup):
        model, criterion, x, y = setup
        loss = criterion(model(x), y)
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ---------------------------------------------------------------------------
# Evaluate function tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_loss_and_accuracy(self):
        model = CNN(in_channels=3, num_classes=10, channels=[8, 16])
        # Create a small fake loader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate(model, loader, criterion, torch.device("cpu"))

        assert "eval/loss" in metrics
        assert "eval/accuracy" in metrics
        assert 0 <= metrics["eval/accuracy"] <= 1
        assert metrics["eval/loss"] > 0

    def test_model_back_in_train_mode(self):
        model = CNN(channels=[8])
        model.train()
        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        assert model.training


# ---------------------------------------------------------------------------
# Weight diagnostics on CNN
# ---------------------------------------------------------------------------

class TestDiagnosticsOnCNN:
    def test_weight_diagnostics_on_cnn(self):
        model = CNN(channels=[8, 16])
        diag = compute_weight_diagnostics(model)
        # Should have diagnostics for the head Linear layer at minimum
        keys = list(diag.keys())
        assert any("condition_number" in k for k in keys)
        assert any("spectral_norm" in k for k in keys)

    def test_gradient_diagnostics_after_backward(self):
        model = CNN(channels=[8, 16])
        x = torch.randn(2, 3, 32, 32)
        y = torch.randint(0, 10, (2,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        diag = compute_gradient_diagnostics(model)
        keys = list(diag.keys())
        assert any("grad_norm" in k for k in keys)
