"""Tests for attention_NMT model, optimizer creation, training, and evaluation."""

import contextlib
import math
import sys
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attention_NMT.model import (
    Transformer,
    MultiHeadAttention,
    EncoderBlock,
    DecoderBlock,
    _make_sinusoidal_pe,
)
from attention_NMT.train import (
    create_optimizer,
    evaluate,
    OptimizerGroup,
    get_lr,
    set_lr,
    setup_precision,
)
from attention_NMT.eval import greedy_decode, compute_translation_metrics
from muon import MuonJordan, MuonLLM
from common.metrics import compute_weight_diagnostics, compute_gradient_diagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128
EMB_DIM = 64
NUM_HEADS = 2  # head_dim = 32
DEPTH = 2
CONTEXT_WINDOW = 32
BATCH_SIZE = 4
PAD_ID = VOCAB_SIZE  # same convention as data.py


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
        pad_id=PAD_ID,
    )
    kw.update(overrides)
    return Transformer(**kw)


def _make_batch(B=BATCH_SIZE, T_src=16, T_tgt=12):
    """Create a fake NMT batch: src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask."""
    src_ids = torch.randint(0, VOCAB_SIZE, (B, T_src))
    tgt_input = torch.randint(0, VOCAB_SIZE, (B, T_tgt))
    tgt_label = torch.randint(0, VOCAB_SIZE, (B, T_tgt))
    src_pad_mask = torch.ones(B, T_src, dtype=torch.bool)
    tgt_pad_mask = torch.ones(B, T_tgt, dtype=torch.bool)
    return src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask


def _infinite_val_loader(B=BATCH_SIZE, T_src=16, T_tgt=12):
    """Infinite generator yielding (src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask)."""
    while True:
        src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = _make_batch(B, T_src, T_tgt)
        yield src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask


# ---------------------------------------------------------------------------
# [CRITICAL] Transformer model tests
# ---------------------------------------------------------------------------

class TestTransformerModel:
    def test_default_construction(self):
        model = _make_model()
        assert isinstance(model, Transformer)

    def test_forward_output_shape(self):
        model = _make_model()
        src_ids, tgt_input, _, src_pad_mask, tgt_pad_mask = _make_batch()
        logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
        B, T_tgt = tgt_input.shape
        num_embeddings = VOCAB_SIZE + 1
        assert logits.shape == (B, T_tgt, num_embeddings)

    def test_forward_without_masks(self):
        """Model should work without padding masks (all-real tokens)."""
        model = _make_model()
        src_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 16))
        tgt_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 12))
        logits = model(src_ids, tgt_input)
        assert logits.shape == (BATCH_SIZE, 12, VOCAB_SIZE + 1)

    def test_different_src_tgt_lengths(self):
        """Encoder and decoder can have different sequence lengths."""
        model = _make_model()
        src_ids, tgt_input, _, src_pad_mask, tgt_pad_mask = _make_batch(
            T_src=20, T_tgt=8
        )
        logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
        assert logits.shape == (BATCH_SIZE, 8, VOCAB_SIZE + 1)

    def test_context_window_assertion(self):
        """Forward should raise if sequence exceeds context window."""
        model = _make_model(context_window=10)
        src_ids = torch.randint(0, VOCAB_SIZE, (2, 15))  # exceeds 10
        tgt_input = torch.randint(0, VOCAB_SIZE, (2, 5))
        with pytest.raises(AssertionError, match="exceeds context window"):
            model(src_ids, tgt_input)

    def test_tgt_context_window_assertion(self):
        model = _make_model(context_window=10)
        src_ids = torch.randint(0, VOCAB_SIZE, (2, 5))
        tgt_input = torch.randint(0, VOCAB_SIZE, (2, 15))  # exceeds 10
        with pytest.raises(AssertionError, match="exceeds context window"):
            model(src_ids, tgt_input)

    def test_pad_id_adds_extra_embedding(self):
        """When pad_id is set, vocab gets +1 for the PAD token."""
        model = _make_model(pad_id=VOCAB_SIZE)
        emb = model.backbone.token_emb
        assert emb.num_embeddings == VOCAB_SIZE + 1
        assert emb.padding_idx == VOCAB_SIZE

    def test_no_pad_id(self):
        """Without pad_id, num_embeddings == vocab_size."""
        model = _make_model(pad_id=None)
        emb = model.backbone.token_emb
        assert emb.num_embeddings == VOCAB_SIZE
        assert emb.padding_idx is None

    def test_backbone_head_split(self):
        model = _make_model()
        assert hasattr(model, "backbone")
        assert hasattr(model, "head")
        assert isinstance(model.backbone, nn.ModuleDict)
        assert isinstance(model.head, nn.Linear)

    def test_encoder_decoder_block_count(self):
        model = _make_model(depth=3)
        assert len(model.backbone.encoder_blocks) == 3
        assert len(model.backbone.decoder_blocks) == 3

    def test_param_count_scales_with_depth(self):
        model_small = _make_model(depth=2)
        model_large = _make_model(depth=4)
        count_small = sum(p.numel() for p in model_small.parameters())
        count_large = sum(p.numel() for p in model_large.parameters())
        assert count_large > count_small

    def test_forward_with_padding(self):
        """Verify model handles padded inputs with proper masks."""
        model = _make_model()
        B, T_src, T_tgt = 2, 16, 12
        src = torch.randint(0, VOCAB_SIZE, (B, T_src))
        tgt = torch.randint(0, VOCAB_SIZE, (B, T_tgt))
        src[:, -4:] = PAD_ID
        tgt[:, -3:] = PAD_ID
        src_mask = (src != PAD_ID)
        tgt_mask = (tgt != PAD_ID)
        logits = model(src, tgt, src_mask, tgt_mask)
        assert logits.shape == (B, T_tgt, VOCAB_SIZE + 1)


# ---------------------------------------------------------------------------
# [HIGH] MultiHeadAttention tests
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_self_attention_output_shape(self):
        mha = MultiHeadAttention(EMB_DIM, NUM_HEADS, dropout=0.0)
        x = torch.randn(BATCH_SIZE, 16, EMB_DIM)
        out = mha(x, x)
        assert out.shape == (BATCH_SIZE, 16, EMB_DIM)

    def test_cross_attention_output_shape(self):
        mha = MultiHeadAttention(EMB_DIM, NUM_HEADS, dropout=0.0)
        q = torch.randn(BATCH_SIZE, 10, EMB_DIM)
        kv = torch.randn(BATCH_SIZE, 20, EMB_DIM)
        out = mha(q, kv)
        assert out.shape == (BATCH_SIZE, 10, EMB_DIM)

    def test_causal_masking(self):
        """With is_causal=True, changing a future token should not affect earlier outputs."""
        mha = MultiHeadAttention(EMB_DIM, NUM_HEADS, dropout=0.0)
        mha.eval()
        x = torch.randn(1, 8, EMB_DIM)
        out_full = mha(x, x, is_causal=True)

        x2 = x.clone()
        x2[0, 7, :] = torch.randn(EMB_DIM)
        out_modified = mha(x2, x2, is_causal=True)

        torch.testing.assert_close(out_full[0, :7], out_modified[0, :7], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# [HIGH] EncoderBlock and DecoderBlock tests
# ---------------------------------------------------------------------------

class TestBlocks:
    def test_encoder_block_residual(self):
        block = EncoderBlock(EMB_DIM, NUM_HEADS, dropout=0.0)
        x = torch.randn(BATCH_SIZE, 16, EMB_DIM)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)

    def test_decoder_block_residual(self):
        block = DecoderBlock(EMB_DIM, NUM_HEADS, dropout=0.0)
        x = torch.randn(BATCH_SIZE, 10, EMB_DIM)
        enc_out = torch.randn(BATCH_SIZE, 16, EMB_DIM)
        out = block(x, enc_out)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)

    def test_decoder_block_uses_encoder_output(self):
        """Changing encoder output should change decoder block output."""
        block = DecoderBlock(EMB_DIM, NUM_HEADS, dropout=0.0)
        block.eval()
        x = torch.randn(BATCH_SIZE, 10, EMB_DIM)
        enc1 = torch.randn(BATCH_SIZE, 16, EMB_DIM)
        enc2 = torch.randn(BATCH_SIZE, 16, EMB_DIM)
        out1 = block(x, enc1)
        out2 = block(x, enc2)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# [HIGH] Sinusoidal positional embedding tests
# ---------------------------------------------------------------------------

class TestSinusoidalPE:
    def test_shape(self):
        pe = _make_sinusoidal_pe(CONTEXT_WINDOW, EMB_DIM)
        assert pe.shape == (1, CONTEXT_WINDOW, EMB_DIM)

    def test_is_buffer_not_parameter(self):
        model = _make_model()
        param_names = {n for n, _ in model.named_parameters()}
        assert not any("pos_emb" in n for n in param_names)
        buffer_names = {n for n, _ in model.named_buffers()}
        assert any("pos_emb" in n for n in buffer_names)

    def test_values_bounded(self):
        pe = _make_sinusoidal_pe(CONTEXT_WINDOW, EMB_DIM)
        assert pe.abs().max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# [CRITICAL] Optimizer creation and parameter splitting
# ---------------------------------------------------------------------------

class TestOptimizerCreation:
    def test_adamw_creates_single_optimizer(self):
        model = _make_model()
        opt = create_optimizer(model, _Args("adamw"))
        assert isinstance(opt, torch.optim.AdamW)

    def test_muon_jordan_creates_optimizer_group(self):
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-jordan"))
        assert isinstance(opt, OptimizerGroup)

    def test_muon_llm_creates_optimizer_group(self):
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-llm"))
        assert isinstance(opt, OptimizerGroup)

    def test_muon_jordan_uses_correct_optimizer_types(self):
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-jordan"))
        types = {type(o) for o in opt.optimizers}
        assert MuonJordan in types
        assert torch.optim.AdamW in types

    def test_muon_llm_uses_correct_optimizer_types(self):
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-llm"))
        types = {type(o) for o in opt.optimizers}
        assert MuonLLM in types
        assert torch.optim.AdamW in types

    def test_muon_param_split_coverage(self):
        """Every parameter must go to exactly one optimizer (Muon or AdamW)."""
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-jordan"))
        total_params = sum(p.numel() for p in model.parameters())

        opt_params = 0
        for sub_opt in opt.optimizers:
            for pg in sub_opt.param_groups:
                opt_params += sum(p.numel() for p in pg["params"])

        assert opt_params == total_params

    def test_muon_only_gets_2d_backbone_linear_weights(self):
        """Muon should only receive 2D backbone Linear .weight tensors."""
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-jordan"))

        muon_opt = opt.optimizers[0]
        assert isinstance(muon_opt, MuonJordan)

        muon_param_ids = set()
        for pg in muon_opt.param_groups:
            for p in pg["params"]:
                muon_param_ids.add(id(p))

        modules = dict(model.named_modules())
        for name, p in model.named_parameters():
            if id(p) in muon_param_ids:
                assert "backbone" in name, f"Muon got non-backbone param: {name}"
                assert name.endswith(".weight"), f"Muon got non-weight param: {name}"
                assert p.ndim == 2, f"Muon got non-2D param: {name} (ndim={p.ndim})"
                mod_name = name.rsplit(".", 1)[0]
                assert isinstance(modules[mod_name], nn.Linear), \
                    f"Muon got non-Linear weight: {name}"

    def test_adam_gets_embeddings_layernorm_head(self):
        """AdamW should get Embedding, LayerNorm, and head params."""
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-jordan"))

        adam_opt = opt.optimizers[1]
        assert isinstance(adam_opt, torch.optim.AdamW)

        adam_param_ids = set()
        for pg in adam_opt.param_groups:
            for p in pg["params"]:
                adam_param_ids.add(id(p))

        assert id(model.backbone.token_emb.weight) in adam_param_ids
        assert id(model.head.weight) in adam_param_ids
        for name, p in model.named_parameters():
            if "ln" in name:
                assert id(p) in adam_param_ids, f"LayerNorm param {name} not in AdamW"


# ---------------------------------------------------------------------------
# [HIGH] OptimizerGroup wrapper tests
# ---------------------------------------------------------------------------

class TestOptimizerGroup:
    def test_state_dict_roundtrip(self):
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-jordan"))
        sd = opt.state_dict()
        assert isinstance(sd, list)
        assert len(sd) == len(opt.optimizers)
        opt.load_state_dict(sd)

    def test_param_groups_aggregation(self):
        model = _make_model()
        opt = create_optimizer(model, _Args("muon-jordan"))
        total_from_group = sum(
            len(pg["params"]) for pg in opt.param_groups
        )
        total_params = len(list(model.parameters()))
        assert total_from_group == total_params


# ---------------------------------------------------------------------------
# [HIGH] LR schedule tests
# ---------------------------------------------------------------------------

class TestLRSchedule:
    def test_warmup_linear_increase(self):
        lr1 = get_lr(0, warmup_steps=10, max_steps=100, max_lr=1.0)
        lr5 = get_lr(5, warmup_steps=10, max_steps=100, max_lr=1.0)
        lr9 = get_lr(9, warmup_steps=10, max_steps=100, max_lr=1.0)
        assert lr1 < lr5 < lr9

    def test_warmup_end_near_max_lr(self):
        lr = get_lr(10, warmup_steps=10, max_steps=100, max_lr=1.0)
        assert lr == pytest.approx(1.0, abs=0.05)

    def test_cosine_decay(self):
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
        src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = _make_batch()
        return model, src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask

    @pytest.mark.parametrize("optim_name", ["adamw", "muon-jordan", "muon-llm"])
    def test_loss_decreases(self, setup, optim_name):
        model, src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = setup
        optimizer = create_optimizer(model, _Args(optim_name, lr=1e-2))

        model.train()
        initial_loss = None
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt_label.view(-1),
                ignore_index=PAD_ID,
            )
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss

    def test_gradients_flow_through_all_params(self, setup):
        model, src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = setup
        logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_label.view(-1),
            ignore_index=PAD_ID,
        )
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    @pytest.mark.parametrize("optim_name", ["muon-jordan", "muon-llm"])
    def test_weight_shapes_preserved_after_muon_step(self, setup, optim_name):
        """Muon NS iteration operates on 2D internally; weights must stay 2D after step."""
        model, src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = setup
        optimizer = create_optimizer(model, _Args(optim_name))

        shapes_before = {n: p.shape for n, p in model.named_parameters()}

        optimizer.zero_grad()
        logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_label.view(-1),
            ignore_index=PAD_ID,
        )
        loss.backward()
        optimizer.step()

        for name, p in model.named_parameters():
            assert p.shape == shapes_before[name], \
                f"{name} shape changed from {shapes_before[name]} to {p.shape}"

    def test_weight_decay_reduces_norms(self, setup):
        model, src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = setup
        optimizer = create_optimizer(model, _Args("muon-jordan", lr=1e-2, weight_decay=0.5))

        initial_norms = {}
        for name, p in model.named_parameters():
            if "weight" in name and p.ndim >= 2:
                initial_norms[name] = p.data.norm().item()

        optimizer.zero_grad()
        logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_label.view(-1),
            ignore_index=PAD_ID,
        )
        loss.backward()
        optimizer.step()

        decreased = sum(
            1 for name, p in model.named_parameters()
            if name in initial_norms and p.data.norm().item() < initial_norms[name]
        )
        assert decreased > 0, "Weight decay did not reduce any weight norms"

    def test_grad_clip(self, setup):
        model, src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = setup
        logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_label.view(-1),
            ignore_index=PAD_ID,
        )
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

        metrics = evaluate(model, val_loader, val_steps=3, pad_id=PAD_ID,
                           precision_ctx=ctx)

        assert "eval/loss" in metrics
        assert "eval/perplexity" in metrics
        assert metrics["eval/loss"] > 0
        assert metrics["eval/perplexity"] > 1.0

    def test_perplexity_consistent_with_loss(self):
        model = _make_model()
        val_loader = _infinite_val_loader()
        ctx = contextlib.nullcontext()

        metrics = evaluate(model, val_loader, val_steps=5, pad_id=PAD_ID,
                           precision_ctx=ctx)

        expected_ppl = math.exp(min(metrics["eval/loss"], 20.0))
        assert metrics["eval/perplexity"] == pytest.approx(expected_ppl, rel=1e-4)

    def test_model_back_in_train_mode(self):
        model = _make_model()
        model.train()
        val_loader = _infinite_val_loader()
        ctx = contextlib.nullcontext()

        evaluate(model, val_loader, val_steps=2, pad_id=PAD_ID, precision_ctx=ctx)

        assert model.training

    def test_no_accuracy_or_regression_keys(self):
        """NMT evaluate returns loss + perplexity only."""
        model = _make_model()
        val_loader = _infinite_val_loader()
        ctx = contextlib.nullcontext()

        metrics = evaluate(model, val_loader, val_steps=2, pad_id=PAD_ID,
                           precision_ctx=ctx)

        assert "eval/accuracy" not in metrics
        assert "eval/mse" not in metrics
        assert "eval/r2" not in metrics

    def test_no_gradients_during_eval(self):
        """evaluate() should run under torch.no_grad()."""
        model = _make_model()
        val_loader = _infinite_val_loader()
        ctx = contextlib.nullcontext()

        evaluate(model, val_loader, val_steps=2, pad_id=PAD_ID, precision_ctx=ctx)

        for name, p in model.named_parameters():
            assert p.grad is None, f"Parameter {name} has gradients after eval"


# ---------------------------------------------------------------------------
# [CRITICAL] Greedy decoding tests
# ---------------------------------------------------------------------------

class TestGreedyDecode:
    def test_returns_list_of_lists(self):
        model = _make_model()
        model.eval()
        B = 2
        src_ids = torch.randint(0, VOCAB_SIZE, (B, 10))
        src_pad_mask = torch.ones(B, 10, dtype=torch.bool)
        precision_ctx = contextlib.nullcontext()

        results = greedy_decode(
            model, src_ids, src_pad_mask,
            bos_id=0, eos_id=1, pad_id=PAD_ID,
            max_len=20, precision_ctx=precision_ctx,
        )

        assert isinstance(results, list)
        assert len(results) == B
        for seq in results:
            assert isinstance(seq, list)

    def test_respects_max_len(self):
        model = _make_model()
        model.eval()
        src_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        src_pad_mask = torch.ones(1, 8, dtype=torch.bool)
        precision_ctx = contextlib.nullcontext()

        results = greedy_decode(
            model, src_ids, src_pad_mask,
            bos_id=0, eos_id=1, pad_id=PAD_ID,
            max_len=5, precision_ctx=precision_ctx,
        )

        for seq in results:
            assert len(seq) <= 5

    def test_output_excludes_eos(self):
        """EOS token should not appear in returned sequences."""
        model = _make_model()
        model.eval()
        src_ids = torch.randint(2, VOCAB_SIZE, (2, 8))
        src_pad_mask = torch.ones(2, 8, dtype=torch.bool)
        precision_ctx = contextlib.nullcontext()
        eos_id = 1

        results = greedy_decode(
            model, src_ids, src_pad_mask,
            bos_id=0, eos_id=eos_id, pad_id=PAD_ID,
            max_len=15, precision_ctx=precision_ctx,
        )

        for seq in results:
            assert eos_id not in seq


# ---------------------------------------------------------------------------
# [HIGH] Translation metrics tests
# ---------------------------------------------------------------------------

class TestTranslationMetrics:
    def test_returns_bleu_and_chrf(self):
        hyps = ["hello world", "this is a test"]
        refs = ["hello world", "this is a test"]
        metrics = compute_translation_metrics(hyps, refs)
        assert "bleu" in metrics
        assert "chrf" in metrics

    def test_perfect_translation_high_bleu(self):
        hyps = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        metrics = compute_translation_metrics(hyps, refs)
        assert metrics["bleu"] > 50

    def test_bad_translation_low_bleu(self):
        hyps = ["xyz abc def ghi"]
        refs = ["the cat sat on the mat"]
        metrics = compute_translation_metrics(hyps, refs)
        assert metrics["bleu"] < 10

    def test_no_comet_by_default(self):
        hyps = ["hello"]
        refs = ["hello"]
        metrics = compute_translation_metrics(hyps, refs)
        assert "comet" not in metrics


# ---------------------------------------------------------------------------
# [HIGH] Weight and gradient diagnostics
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
        src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = _make_batch()
        logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_label.view(-1),
            ignore_index=PAD_ID,
        )
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
        assert any("head" in k for k in diag.keys())

    def test_diagnostics_covers_attention_projections(self):
        model = _make_model()
        diag = compute_weight_diagnostics(model)
        assert any("q_proj" in k for k in diag.keys())
        assert any("k_proj" in k for k in diag.keys())
        assert any("v_proj" in k for k in diag.keys())
        assert any("out_proj" in k for k in diag.keys())

    def test_diagnostics_covers_cross_attention(self):
        """NMT model has cross-attention in decoder -- diagnostics should include it."""
        model = _make_model()
        diag = compute_weight_diagnostics(model)
        assert any("cross_attn" in k for k in diag.keys())

    def test_diagnostics_values_finite(self):
        model = _make_model()
        diag = compute_weight_diagnostics(model)
        for k, v in diag.items():
            if isinstance(v, (int, float)):
                assert math.isfinite(v), f"Non-finite value for {k}: {v}"
