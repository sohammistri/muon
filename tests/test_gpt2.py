"""Tests for GPT2 model, optimizer creation, training, and evaluation."""

import contextlib
import math
import sys
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from GPT2.model import GPT2, CausalSelfAttention, TransformerBlock, _make_sinusoidal_pe
from GPT2.train import (
    create_optimizer,
    evaluate,
    OptimizerGroup,
    get_lr,
    set_lr,
    setup_precision,
)
from muon import MuonJordan, MuonLLM
from common.metrics import compute_weight_diagnostics, compute_gradient_diagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small model defaults for all tests
VOCAB_SIZE = 128
EMB_DIM = 64
NUM_HEADS = 2  # head_dim = 32
DEPTH = 2
CONTEXT_WINDOW = 32
BATCH_SIZE = 4


class _Args:
    """Minimal args namespace for create_optimizer."""
    def __init__(self, optim, lr=1e-3, weight_decay=1e-4):
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay


class _PrecisionArgs:
    """Args namespace for setup_precision."""
    def __init__(self, precision="fp32"):
        self.precision = precision


def _make_model(**overrides):
    kw = dict(
        vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, num_heads=NUM_HEADS,
        depth=DEPTH, context_window=CONTEXT_WINDOW, dropout=0.0,
    )
    kw.update(overrides)
    return GPT2(**kw)


def _make_batch(B=BATCH_SIZE, T=CONTEXT_WINDOW):
    inputs = torch.randint(0, VOCAB_SIZE, (B, T))
    targets = torch.randint(0, VOCAB_SIZE, (B, T))
    return inputs, targets


def _infinite_val_loader(B=BATCH_SIZE, T=CONTEXT_WINDOW):
    """Infinite generator yielding (inputs, targets) for evaluate()."""
    while True:
        yield _make_batch(B, T)


# ---------------------------------------------------------------------------
# [CRITICAL] GPT2 Model tests
# ---------------------------------------------------------------------------

class TestGPT2Model:
    def test_default_construction(self):
        model = _make_model()
        assert isinstance(model.backbone, nn.ModuleDict)
        assert isinstance(model.backbone.token_emb, nn.Embedding)
        assert isinstance(model.backbone.blocks, nn.ModuleList)
        assert isinstance(model.backbone.ln_f, nn.LayerNorm)
        assert isinstance(model.head, nn.Linear)

    def test_head_output_dim_matches_vocab(self):
        model = _make_model()
        assert model.head.out_features == VOCAB_SIZE
        assert model.head.in_features == EMB_DIM

    def test_custom_config(self):
        model = _make_model(vocab_size=256, emb_dim=128, num_heads=4, depth=4, context_window=64)
        assert model.head.out_features == 256
        assert model.head.in_features == 128
        assert len(model.backbone.blocks) == 4
        assert model.context_window == 64

    def test_forward_shape(self):
        model = _make_model()
        x, _ = _make_batch()
        out = model(x)
        assert out.shape == (BATCH_SIZE, CONTEXT_WINDOW, VOCAB_SIZE)

    def test_forward_shorter_sequence(self):
        """Sequences shorter than context_window should work."""
        model = _make_model()
        x = torch.randint(0, VOCAB_SIZE, (2, 10))
        out = model(x)
        assert out.shape == (2, 10, VOCAB_SIZE)

    def test_forward_rejects_too_long_sequence(self):
        model = _make_model(context_window=16)
        x = torch.randint(0, VOCAB_SIZE, (1, 32))
        with pytest.raises(AssertionError, match="exceeds context window"):
            model(x)

    def test_depth_determines_block_count(self):
        for d in [1, 3, 6]:
            model = _make_model(depth=d)
            assert len(model.backbone.blocks) == d

    def test_parameter_count_scales_with_depth(self):
        small = _make_model(depth=2)
        large = _make_model(depth=6)
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params

    def test_parameter_count_scales_with_emb_dim(self):
        small = _make_model(emb_dim=32, num_heads=1)
        large = _make_model(emb_dim=128, num_heads=4)
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params

    def test_sinusoidal_pe_shape(self):
        pe = _make_sinusoidal_pe(64, 32)
        assert pe.shape == (1, 64, 32)

    def test_sinusoidal_pe_not_all_zeros(self):
        pe = _make_sinusoidal_pe(64, 32)
        assert pe.abs().sum() > 0

    def test_pos_emb_is_buffer_not_parameter(self):
        """Positional embedding should be a buffer, not a learnable parameter."""
        model = _make_model()
        param_names = {n for n, _ in model.named_parameters()}
        assert "backbone.pos_emb" not in param_names
        assert "pos_emb" in dict(model.backbone.named_buffers())


# ---------------------------------------------------------------------------
# [HIGH] CausalSelfAttention tests
# ---------------------------------------------------------------------------

class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(emb_dim=64, num_heads=2, dropout=0.0)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_masking(self):
        """Output at position t should not depend on inputs at position t+1."""
        attn = CausalSelfAttention(emb_dim=64, num_heads=2, dropout=0.0)
        attn.eval()
        x = torch.randn(1, 8, 64)

        out_full = attn(x)
        # Run with only the first 4 tokens
        out_prefix = attn(x[:, :4, :])

        # Outputs for the first 4 positions should be the same
        torch.testing.assert_close(out_full[:, :4, :], out_prefix, atol=1e-5, rtol=1e-5)

    def test_emb_dim_not_divisible_by_heads_raises(self):
        with pytest.raises(AssertionError):
            CausalSelfAttention(emb_dim=65, num_heads=2)


# ---------------------------------------------------------------------------
# [HIGH] TransformerBlock tests
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(emb_dim=64, num_heads=2, dropout=0.0)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (not identity) but not be wildly different."""
        block = TransformerBlock(emb_dim=64, num_heads=2, dropout=0.0)
        block.eval()
        x = torch.randn(1, 8, 64)
        out = block(x)
        # Not identical (residual + FFN does something)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# [CRITICAL] Optimizer creation tests
# ---------------------------------------------------------------------------

class TestOptimizerCreation:
    @pytest.fixture
    def model(self):
        return _make_model()

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

    def test_muon_param_split_completeness(self, model):
        """All parameters must be covered: muon_count + adam_count == total."""
        opt = create_optimizer(model, _Args("muon-jordan"))
        muon_opt = opt.optimizers[0]
        adam_opt = opt.optimizers[1]

        muon_count = sum(p.numel() for g in muon_opt.param_groups for p in g["params"])
        adam_count = sum(p.numel() for g in adam_opt.param_groups for p in g["params"])
        total = sum(p.numel() for p in model.parameters())

        assert muon_count + adam_count == total
        assert muon_count > 0
        assert adam_count > 0

    def test_muon_params_are_2d_linear_weights(self, model):
        """Muon should only receive 2D backbone Linear weights."""
        opt = create_optimizer(model, _Args("muon-jordan"))
        muon_opt = opt.optimizers[0]

        for g in muon_opt.param_groups:
            for p in g["params"]:
                assert p.ndim == 2, \
                    f"Muon should only receive 2D Linear weights, got ndim={p.ndim}"

    def test_muon_excludes_embedding_and_head(self, model):
        """Embedding, LayerNorm, and head params must NOT go to Muon."""
        opt = create_optimizer(model, _Args("muon-llm"))
        muon_opt = opt.optimizers[0]

        muon_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}

        # Embedding weight must not be in Muon
        assert id(model.backbone.token_emb.weight) not in muon_ids
        # Head weight must not be in Muon
        assert id(model.head.weight) not in muon_ids
        # LayerNorm weights must not be in Muon
        assert id(model.backbone.ln_f.weight) not in muon_ids

    def test_muon_includes_backbone_linear_weights(self, model):
        """Backbone Linear weights (qkv_proj, out_proj, ffn) must go to Muon."""
        opt = create_optimizer(model, _Args("muon-jordan"))
        muon_opt = opt.optimizers[0]
        muon_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}

        block = model.backbone.blocks[0]
        assert id(block.attn.qkv_proj.weight) in muon_ids
        assert id(block.attn.out_proj.weight) in muon_ids
        assert id(block.ffn[0].weight) in muon_ids  # ffn up-projection
        assert id(block.ffn[2].weight) in muon_ids  # ffn down-projection

    def test_unknown_optim_falls_through_to_muon(self, model):
        """create_optimizer only checks for 'adamw' and 'muon-jordan'; else → MuonLLM.
        CLI parse_args restricts valid choices, so this documents the fallback behavior."""
        opt = create_optimizer(model, _Args("anything-else"))
        assert isinstance(opt, OptimizerGroup)
        assert isinstance(opt.optimizers[0], MuonLLM)


# ---------------------------------------------------------------------------
# [HIGH] OptimizerGroup tests
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

    def test_param_groups_aggregates_all(self):
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(3))
        opt1 = torch.optim.SGD([p1], lr=0.1)
        opt2 = torch.optim.Adam([p2], lr=0.01)
        group = OptimizerGroup(opt1, opt2)

        # Should aggregate param_groups from both optimizers
        assert len(group.param_groups) == 2


# ---------------------------------------------------------------------------
# [CRITICAL] LR schedule tests
# ---------------------------------------------------------------------------

class TestLRSchedule:
    def test_warmup_linear_ramp(self):
        """During warmup, LR should increase linearly."""
        lr0 = get_lr(0, warmup_steps=10, max_steps=100, max_lr=1.0)
        lr5 = get_lr(5, warmup_steps=10, max_steps=100, max_lr=1.0)
        lr9 = get_lr(9, warmup_steps=10, max_steps=100, max_lr=1.0)
        assert lr0 < lr5 < lr9
        assert lr0 == pytest.approx(0.1, abs=1e-6)  # (0+1)/10
        assert lr5 == pytest.approx(0.6, abs=1e-6)  # (5+1)/10

    def test_peak_lr_at_warmup_boundary(self):
        """At warmup_steps, LR should be at or near max_lr."""
        lr = get_lr(10, warmup_steps=10, max_steps=100, max_lr=1.0)
        assert lr == pytest.approx(1.0, abs=0.05)

    def test_cosine_decay(self):
        """After warmup, LR should decrease via cosine."""
        lr_early = get_lr(20, warmup_steps=10, max_steps=100, max_lr=1.0)
        lr_mid = get_lr(50, warmup_steps=10, max_steps=100, max_lr=1.0)
        lr_late = get_lr(90, warmup_steps=10, max_steps=100, max_lr=1.0)
        assert lr_early > lr_mid > lr_late

    def test_lr_at_max_steps_is_min(self):
        lr = get_lr(100, warmup_steps=10, max_steps=100, max_lr=1.0, min_lr=0.0)
        assert lr == pytest.approx(0.0, abs=1e-6)

    def test_set_lr_on_regular_optimizer(self):
        model = _make_model()
        opt = create_optimizer(model, _Args("adamw", lr=1e-3))
        set_lr(opt, 5e-4)
        for pg in opt.param_groups:
            assert pg["lr"] == 5e-4

    def test_set_lr_on_optimizer_group(self):
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-jordan", lr=1e-3))
        set_lr(opt, 2e-4)
        for pg in opt.param_groups:
            assert pg["lr"] == 2e-4


# ---------------------------------------------------------------------------
# [HIGH] Precision setup tests
# ---------------------------------------------------------------------------

class TestPrecision:
    def test_fp32_returns_nullcontext(self):
        model = _make_model()
        ctx = setup_precision(_PrecisionArgs("fp32"), model, "cpu")
        assert isinstance(ctx, contextlib.nullcontext)

    def test_bf16_returns_autocast(self):
        """bf16 should return an autocast context (test on CPU)."""
        model = _make_model()
        ctx = setup_precision(_PrecisionArgs("bf16"), model, "cpu")
        assert isinstance(ctx, torch.amp.autocast)


# ---------------------------------------------------------------------------
# [CRITICAL] Training step tests
# ---------------------------------------------------------------------------

class TestTrainingStep:
    @pytest.fixture
    def setup(self):
        model = _make_model()
        x, y = _make_batch()
        return model, x, y

    @pytest.mark.parametrize("optim_name", ["adamw", "muon-jordan", "muon-llm"])
    def test_loss_decreases(self, setup, optim_name):
        model, x, y = setup
        optimizer = create_optimizer(model, _Args(optim_name, lr=1e-2))

        model.train()
        initial_loss = None
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss

    def test_gradients_flow_through_all_params(self, setup):
        model, x, y = setup
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    @pytest.mark.parametrize("optim_name", ["muon-jordan", "muon-llm"])
    def test_weight_shapes_preserved_after_muon_step(self, setup, optim_name):
        """Muon NS iteration operates on 2D internally; weights must stay 2D after step."""
        model, x, y = setup
        optimizer = create_optimizer(model, _Args(optim_name))

        # Record shapes before
        shapes_before = {n: p.shape for n, p in model.named_parameters()}

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        # All shapes must be preserved
        for name, p in model.named_parameters():
            assert p.shape == shapes_before[name], \
                f"{name} shape changed from {shapes_before[name]} to {p.shape}"

    def test_weight_decay_reduces_norms(self, setup):
        model, x, y = setup
        optimizer = create_optimizer(model, _Args("muon-jordan", lr=1e-2, weight_decay=0.5))

        initial_norms = {}
        for name, p in model.named_parameters():
            if "weight" in name and p.ndim >= 2:
                initial_norms[name] = p.data.norm().item()

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        decreased = sum(
            1 for name, p in model.named_parameters()
            if name in initial_norms and p.data.norm().item() < initial_norms[name]
        )
        assert decreased > 0, "Weight decay did not reduce any weight norms"

    def test_grad_clip(self, setup):
        """After clipping, global grad norm should not exceed the clip value."""
        model, x, y = setup
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        total_norm = torch.norm(
            torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
        ).item()
        assert total_norm <= max_norm + 1e-3


# ---------------------------------------------------------------------------
# [CRITICAL] Evaluate function tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_loss_and_perplexity(self):
        model = _make_model()
        val_loader = _infinite_val_loader()
        ctx = contextlib.nullcontext()

        metrics = evaluate(model, val_loader, val_steps=3, precision_ctx=ctx)

        assert "eval/loss" in metrics
        assert "eval/perplexity" in metrics
        assert metrics["eval/loss"] > 0
        assert metrics["eval/perplexity"] > 1.0  # exp(loss) > 1 for loss > 0

    def test_perplexity_consistent_with_loss(self):
        model = _make_model()
        val_loader = _infinite_val_loader()
        ctx = contextlib.nullcontext()

        metrics = evaluate(model, val_loader, val_steps=5, precision_ctx=ctx)

        expected_ppl = math.exp(min(metrics["eval/loss"], 20.0))
        assert metrics["eval/perplexity"] == pytest.approx(expected_ppl, rel=1e-4)

    def test_model_back_in_train_mode(self):
        model = _make_model()
        model.train()
        val_loader = _infinite_val_loader()
        ctx = contextlib.nullcontext()

        evaluate(model, val_loader, val_steps=2, precision_ctx=ctx)

        assert model.training

    def test_no_accuracy_or_regression_keys(self):
        """GPT2 evaluate returns loss + perplexity only, not accuracy/mse/r2."""
        model = _make_model()
        val_loader = _infinite_val_loader()
        ctx = contextlib.nullcontext()

        metrics = evaluate(model, val_loader, val_steps=2, precision_ctx=ctx)

        assert "eval/accuracy" not in metrics
        assert "eval/mse" not in metrics
        assert "eval/r2" not in metrics

    def test_no_gradients_during_eval(self):
        """evaluate() should run under torch.no_grad()."""
        model = _make_model()
        val_loader = _infinite_val_loader()
        ctx = contextlib.nullcontext()

        evaluate(model, val_loader, val_steps=2, precision_ctx=ctx)

        # No parameter should have accumulated gradients from eval
        for name, p in model.named_parameters():
            assert p.grad is None, f"Parameter {name} has gradients after eval"


# ---------------------------------------------------------------------------
# [HIGH] Weight and gradient diagnostics on GPT2
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_weight_diagnostics_keys(self):
        model = _make_model()
        diag = compute_weight_diagnostics(model)
        keys = list(diag.keys())
        assert any("condition_number" in k for k in keys)
        assert any("spectral_norm" in k for k in keys)
        assert any("orthogonality_error" in k for k in keys)
        assert any("effective_rank" in k for k in keys)

    def test_gradient_diagnostics_after_backward(self):
        model = _make_model()
        x, y = _make_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        diag = compute_gradient_diagnostics(model)
        keys = list(diag.keys())
        assert any("grad_norm" in k for k in keys)

    def test_diagnostics_keys_have_correct_prefix(self):
        model = _make_model()
        diag = compute_weight_diagnostics(model)
        for k in diag.keys():
            assert k.startswith("diagnostics/"), f"Key {k} missing diagnostics/ prefix"

    def test_diagnostics_covers_head_layer(self):
        model = _make_model()
        diag = compute_weight_diagnostics(model)
        assert any("head" in k for k in diag.keys()), \
            "Weight diagnostics should include the head Linear layer"

    def test_diagnostics_covers_backbone_attention(self):
        model = _make_model()
        diag = compute_weight_diagnostics(model)
        assert any("qkv_proj" in k for k in diag.keys()), \
            "Weight diagnostics should include attention qkv_proj"
        assert any("out_proj" in k for k in diag.keys()), \
            "Weight diagnostics should include attention out_proj"

    def test_diagnostics_values_finite(self):
        model = _make_model()
        diag = compute_weight_diagnostics(model)
        for k, v in diag.items():
            if isinstance(v, (int, float)):
                assert math.isfinite(v), f"Non-finite value for {k}: {v}"
