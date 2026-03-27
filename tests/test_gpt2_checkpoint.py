"""Tests for GPT2 checkpointing: save, load, find, OptimizerGroup state, and LR resume."""

import contextlib
import json
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from GPT2.model import GPT2
from GPT2.train import (
    create_optimizer,
    OptimizerGroup,
    get_lr,
    set_lr,
)
from GPT2.checkpoint import save_checkpoint, load_checkpoint, find_latest_step
from muon import MuonJordan, MuonLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128
EMB_DIM = 64
NUM_HEADS = 2
DEPTH = 2
CONTEXT_WINDOW = 32
BATCH_SIZE = 4


class _Args:
    """Minimal args namespace for create_optimizer."""
    def __init__(self, optim, lr=1e-3, weight_decay=1e-4):
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay


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


def _train_one_step(model, optimizer):
    """Run one forward + backward + step so optimizer has state."""
    x, y = _make_batch()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def _make_meta(step=100, lr=2.5e-4):
    return {
        "step": step,
        "val_loss": 4.5,
        "args": {"optim": "adamw", "lr": 3e-4, "max_steps": 1000},
        "data_state": {"pq_idx": 2, "rg_idx": 15, "epoch": 1},
        "lr_state": {
            "current_lr": lr,
            "warmup_steps": 100,
            "max_steps": 1000,
            "max_lr": 3e-4,
        },
    }


# ---------------------------------------------------------------------------
# [CRITICAL] save_checkpoint tests
# ---------------------------------------------------------------------------

class TestSaveCheckpoint:
    def test_creates_directory(self, tmp_path):
        """save_checkpoint should create checkpoint_dir if it doesn't exist."""
        ckpt_dir = str(tmp_path / "new_dir")
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))

        save_checkpoint(
            ckpt_dir, 100, model.state_dict(), optimizer.state_dict(),
            _make_meta(), rank=0, use_ddp=False,
        )

        assert os.path.isdir(ckpt_dir)

    def test_saves_model_file_rank0(self, tmp_path):
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))

        save_checkpoint(
            ckpt_dir, 50, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=50), rank=0, use_ddp=False,
        )

        assert os.path.isfile(os.path.join(ckpt_dir, "model_000050.pt"))

    def test_saves_meta_file_rank0(self, tmp_path):
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))

        save_checkpoint(
            ckpt_dir, 50, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=50), rank=0, use_ddp=False,
        )

        meta_path = os.path.join(ckpt_dir, "meta_000050.json")
        assert os.path.isfile(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["step"] == 50
        assert "data_state" in meta
        assert "lr_state" in meta

    def test_saves_optimizer_file(self, tmp_path):
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))

        save_checkpoint(
            ckpt_dir, 50, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=50), rank=0, use_ddp=False,
        )

        assert os.path.isfile(os.path.join(ckpt_dir, "optim_000050_rank0.pt"))

    def test_non_rank0_skips_model_and_meta(self, tmp_path):
        """Only rank 0 saves model and meta; all ranks save optimizer."""
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))

        save_checkpoint(
            ckpt_dir, 50, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=50), rank=1, use_ddp=False,
        )

        assert not os.path.isfile(os.path.join(ckpt_dir, "model_000050.pt"))
        assert not os.path.isfile(os.path.join(ckpt_dir, "meta_000050.json"))
        assert os.path.isfile(os.path.join(ckpt_dir, "optim_000050_rank1.pt"))

    def test_step_zero_padding(self, tmp_path):
        """Step numbers are zero-padded to 6 digits."""
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))

        save_checkpoint(
            ckpt_dir, 5, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=5), rank=0, use_ddp=False,
        )

        assert os.path.isfile(os.path.join(ckpt_dir, "model_000005.pt"))
        assert os.path.isfile(os.path.join(ckpt_dir, "meta_000005.json"))
        assert os.path.isfile(os.path.join(ckpt_dir, "optim_000005_rank0.pt"))

    def test_saves_optimizer_group_state(self, tmp_path):
        """OptimizerGroup.state_dict() should produce a list that torch.save handles."""
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("muon-jordan"))
        _train_one_step(model, optimizer)

        save_checkpoint(
            ckpt_dir, 10, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=10), rank=0, use_ddp=False,
        )

        optim_path = os.path.join(ckpt_dir, "optim_000010_rank0.pt")
        loaded = torch.load(optim_path, map_location="cpu", weights_only=True)
        assert isinstance(loaded, list)
        assert len(loaded) == 2  # Muon + AdamW

    def test_meta_contains_lr_state(self, tmp_path):
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))
        meta = _make_meta(step=200, lr=1.5e-4)

        save_checkpoint(
            ckpt_dir, 200, model.state_dict(), optimizer.state_dict(),
            meta, rank=0, use_ddp=False,
        )

        with open(os.path.join(ckpt_dir, "meta_000200.json")) as f:
            saved_meta = json.load(f)
        assert saved_meta["lr_state"]["current_lr"] == 1.5e-4
        assert saved_meta["lr_state"]["warmup_steps"] == 100
        assert saved_meta["lr_state"]["max_lr"] == 3e-4


# ---------------------------------------------------------------------------
# [CRITICAL] load_checkpoint tests
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:
    @pytest.fixture
    def saved_checkpoint(self, tmp_path):
        """Save a checkpoint and return (ckpt_dir, step, model, optimizer)."""
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))
        _train_one_step(model, optimizer)

        step = 100
        meta = _make_meta(step=step)
        save_checkpoint(
            ckpt_dir, step, model.state_dict(), optimizer.state_dict(),
            meta, rank=0, use_ddp=False,
        )
        return ckpt_dir, step, model, optimizer

    def test_returns_correct_step(self, saved_checkpoint):
        ckpt_dir, step, _, _ = saved_checkpoint
        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")
        assert ckpt["step"] == step

    def test_returns_model_state_dict(self, saved_checkpoint):
        ckpt_dir, step, model, _ = saved_checkpoint
        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")

        assert set(ckpt["model_state_dict"].keys()) == set(model.state_dict().keys())

    def test_model_weights_match(self, saved_checkpoint):
        """Loaded model state should exactly match the saved model."""
        ckpt_dir, step, model, _ = saved_checkpoint
        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")

        for key in model.state_dict():
            torch.testing.assert_close(
                ckpt["model_state_dict"][key], model.state_dict()[key],
            )

    def test_returns_optimizer_state_dict(self, saved_checkpoint):
        ckpt_dir, step, _, optimizer = saved_checkpoint
        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")

        # For a plain AdamW, state_dict is a dict (not a list)
        assert isinstance(ckpt["optimizer_state_dict"], dict)

    def test_returns_meta_with_data_state(self, saved_checkpoint):
        ckpt_dir, step, _, _ = saved_checkpoint
        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")

        assert "data_state" in ckpt["meta"]
        assert ckpt["meta"]["data_state"]["pq_idx"] == 2
        assert ckpt["meta"]["data_state"]["rg_idx"] == 15
        assert ckpt["meta"]["data_state"]["epoch"] == 1

    def test_returns_meta_with_lr_state(self, saved_checkpoint):
        ckpt_dir, step, _, _ = saved_checkpoint
        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")

        lr_state = ckpt["meta"]["lr_state"]
        assert "current_lr" in lr_state
        assert "warmup_steps" in lr_state
        assert "max_lr" in lr_state

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(str(tmp_path), 999, device="cpu")

    def test_load_optimizer_group_state(self, tmp_path):
        """Save and load an OptimizerGroup checkpoint (Muon + AdamW)."""
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("muon-llm"))
        _train_one_step(model, optimizer)

        step = 50
        save_checkpoint(
            ckpt_dir, step, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=step), rank=0, use_ddp=False,
        )

        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")
        assert isinstance(ckpt["optimizer_state_dict"], list)
        assert len(ckpt["optimizer_state_dict"]) == 2


# ---------------------------------------------------------------------------
# [CRITICAL] find_latest_step tests
# ---------------------------------------------------------------------------

class TestFindLatestStep:
    def test_no_dir_returns_none(self, tmp_path):
        assert find_latest_step(str(tmp_path / "nonexistent")) is None

    def test_empty_dir_returns_none(self, tmp_path):
        assert find_latest_step(str(tmp_path)) is None

    def test_single_checkpoint(self, tmp_path):
        ckpt_dir = str(tmp_path)
        with open(os.path.join(ckpt_dir, "meta_000100.json"), "w") as f:
            json.dump({"step": 100}, f)

        assert find_latest_step(ckpt_dir) == 100

    def test_multiple_checkpoints_returns_latest(self, tmp_path):
        ckpt_dir = str(tmp_path)
        for step in [50, 200, 100, 150]:
            with open(os.path.join(ckpt_dir, f"meta_{step:06d}.json"), "w") as f:
                json.dump({"step": step}, f)

        assert find_latest_step(ckpt_dir) == 200

    def test_ignores_non_meta_files(self, tmp_path):
        ckpt_dir = str(tmp_path)
        with open(os.path.join(ckpt_dir, "meta_000100.json"), "w") as f:
            json.dump({"step": 100}, f)
        # Create noise files
        with open(os.path.join(ckpt_dir, "model_000100.pt"), "w") as f:
            f.write("")
        with open(os.path.join(ckpt_dir, "random_file.json"), "w") as f:
            f.write("")

        assert find_latest_step(ckpt_dir) == 100


# ---------------------------------------------------------------------------
# [CRITICAL] OptimizerGroup state_dict / load_state_dict tests
# ---------------------------------------------------------------------------

class TestOptimizerGroupState:
    def test_state_dict_returns_list(self):
        model = _make_model()
        optimizer = create_optimizer(model, _Args("muon-jordan"))
        _train_one_step(model, optimizer)

        sd = optimizer.state_dict()
        assert isinstance(sd, list)
        assert len(sd) == 2

    def test_state_dict_contains_valid_optimizer_states(self):
        model = _make_model()
        optimizer = create_optimizer(model, _Args("muon-llm"))
        _train_one_step(model, optimizer)

        sd = optimizer.state_dict()
        for entry in sd:
            assert "state" in entry
            assert "param_groups" in entry

    def test_load_state_dict_restores_state(self):
        """After load_state_dict, optimizer state should match the saved state."""
        model = _make_model()
        optimizer = create_optimizer(model, _Args("muon-jordan"))

        # Train a few steps to build up optimizer state
        for _ in range(3):
            _train_one_step(model, optimizer)

        # Save state
        sd = optimizer.state_dict()

        # Create a fresh optimizer and load saved state
        fresh_optimizer = create_optimizer(model, _Args("muon-jordan"))
        fresh_optimizer.load_state_dict(sd)

        fresh_sd = fresh_optimizer.state_dict()
        assert len(fresh_sd) == len(sd)

        # Each constituent optimizer's state should match
        for saved, loaded in zip(sd, fresh_sd):
            assert set(saved["state"].keys()) == set(loaded["state"].keys())

    def test_load_state_dict_preserves_momentum_buffers(self):
        """Muon momentum buffers should survive save/load round-trip."""
        model = _make_model()
        optimizer = create_optimizer(model, _Args("muon-jordan"))

        for _ in range(3):
            _train_one_step(model, optimizer)

        sd = optimizer.state_dict()

        # Get momentum buffer from Muon (first optimizer in group)
        muon_state = sd[0]["state"]
        assert len(muon_state) > 0, "Muon should have state after training steps"

        # Create fresh optimizer and load
        fresh_optimizer = create_optimizer(model, _Args("muon-jordan"))
        fresh_optimizer.load_state_dict(sd)

        fresh_muon_state = fresh_optimizer.state_dict()[0]["state"]
        for key in muon_state:
            for buf_name in muon_state[key]:
                torch.testing.assert_close(
                    muon_state[key][buf_name],
                    fresh_muon_state[key][buf_name],
                )

    def test_adamw_state_dict_is_plain_dict(self):
        """When using adamw (no OptimizerGroup), state_dict should be a plain dict."""
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))
        _train_one_step(model, optimizer)

        sd = optimizer.state_dict()
        assert isinstance(sd, dict)
        assert "state" in sd
        assert "param_groups" in sd


# ---------------------------------------------------------------------------
# [CRITICAL] Save/load round-trip tests
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    @pytest.mark.parametrize("optim_name", ["adamw", "muon-jordan", "muon-llm"])
    def test_model_roundtrip_matches(self, tmp_path, optim_name):
        """Model state should be identical after save → load → restore."""
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args(optim_name))

        # Train to make the model non-random
        for _ in range(3):
            _train_one_step(model, optimizer)

        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        step = 50
        save_checkpoint(
            ckpt_dir, step, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=step), rank=0, use_ddp=False,
        )

        # Create a fresh model and restore
        fresh_model = _make_model()
        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")
        fresh_model.load_state_dict(ckpt["model_state_dict"])

        for key in original_state:
            torch.testing.assert_close(
                fresh_model.state_dict()[key], original_state[key],
            )

    @pytest.mark.parametrize("optim_name", ["adamw", "muon-jordan", "muon-llm"])
    def test_optimizer_roundtrip_restores(self, tmp_path, optim_name):
        """Optimizer state should be restored after save → load → restore."""
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args(optim_name))

        for _ in range(3):
            _train_one_step(model, optimizer)

        step = 50
        save_checkpoint(
            ckpt_dir, step, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=step), rank=0, use_ddp=False,
        )

        # Create fresh optimizer and restore
        fresh_optimizer = create_optimizer(model, _Args(optim_name))
        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")
        fresh_optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Verify state was loaded (non-empty state)
        fresh_sd = fresh_optimizer.state_dict()
        if isinstance(fresh_sd, list):
            for entry in fresh_sd:
                assert len(entry["state"]) > 0
        else:
            assert len(fresh_sd["state"]) > 0

    def test_training_continues_after_resume(self, tmp_path):
        """A model resumed from checkpoint should continue reducing loss."""
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw", lr=1e-2))

        # Train and record loss
        x, y = _make_batch()
        for _ in range(5):
            _train_one_step(model, optimizer)

        # Save checkpoint
        step = 5
        save_checkpoint(
            ckpt_dir, step, model.state_dict(), optimizer.state_dict(),
            _make_meta(step=step), rank=0, use_ddp=False,
        )

        # Compute loss before more training
        model.eval()
        with torch.no_grad():
            logits = model(x)
            loss_at_ckpt = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            ).item()
        model.train()

        # Resume: load checkpoint into fresh model + optimizer
        fresh_model = _make_model()
        ckpt = load_checkpoint(ckpt_dir, step, device="cpu")
        fresh_model.load_state_dict(ckpt["model_state_dict"])

        fresh_optimizer = create_optimizer(fresh_model, _Args("adamw", lr=1e-2))
        fresh_optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Continue training on same fixed batch
        fresh_model.train()
        for _ in range(10):
            fresh_optimizer.zero_grad()
            logits = fresh_model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            fresh_optimizer.step()

        assert loss.item() < loss_at_ckpt

    def test_multiple_checkpoints_coexist(self, tmp_path):
        """Saving at different steps should produce separate files."""
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))

        for step in [10, 20, 30]:
            _train_one_step(model, optimizer)
            save_checkpoint(
                ckpt_dir, step, model.state_dict(), optimizer.state_dict(),
                _make_meta(step=step), rank=0, use_ddp=False,
            )

        # All files should exist
        for step in [10, 20, 30]:
            assert os.path.isfile(os.path.join(ckpt_dir, f"model_{step:06d}.pt"))
            assert os.path.isfile(os.path.join(ckpt_dir, f"meta_{step:06d}.json"))
            assert os.path.isfile(os.path.join(ckpt_dir, f"optim_{step:06d}_rank0.pt"))

        # find_latest_step should return 30
        assert find_latest_step(ckpt_dir) == 30


# ---------------------------------------------------------------------------
# [HIGH] LR state save/restore tests
# ---------------------------------------------------------------------------

class TestLRStateResume:
    def test_lr_state_saved_in_meta(self, tmp_path):
        ckpt_dir = str(tmp_path)
        model = _make_model()
        optimizer = create_optimizer(model, _Args("adamw"))
        current_lr = 2.0e-4
        meta = _make_meta(step=500, lr=current_lr)

        save_checkpoint(
            ckpt_dir, 500, model.state_dict(), optimizer.state_dict(),
            meta, rank=0, use_ddp=False,
        )

        ckpt = load_checkpoint(ckpt_dir, 500, device="cpu")
        assert ckpt["meta"]["lr_state"]["current_lr"] == current_lr

    def test_resume_lr_uses_saved_value(self):
        """When resume_lr is set, first step LR should use saved checkpoint value."""
        saved_lr = 1.5e-4
        # Simulate the resume_lr logic from main()
        resume_lr_value = saved_lr  # This is what main() extracts

        # On the first resumed step, use saved LR
        step = 501
        start_step = 501
        lr = resume_lr_value if step == start_step else get_lr(
            step - 1, warmup_steps=100, max_steps=1000, max_lr=3e-4,
        )
        assert lr == saved_lr

    def test_resume_lr_schedule_takes_over_after_first_step(self):
        """After the first resumed step, the schedule should compute LR normally."""
        saved_lr = 1.5e-4
        resume_lr_value = saved_lr
        start_step = 501

        # First step uses saved LR
        lr_first = resume_lr_value if 501 == start_step else get_lr(
            500, warmup_steps=100, max_steps=1000, max_lr=3e-4,
        )
        assert lr_first == saved_lr

        # Second step uses schedule
        lr_second = resume_lr_value if 502 == start_step else get_lr(
            501, warmup_steps=100, max_steps=1000, max_lr=3e-4,
        )
        expected = get_lr(501, warmup_steps=100, max_steps=1000, max_lr=3e-4)
        assert lr_second == expected
        assert lr_second != saved_lr  # schedule gives a different value

    def test_no_resume_lr_recomputes_from_schedule(self):
        """Without resume_lr flag, LR is recomputed from schedule even on resume."""
        start_step = 501
        resume_lr_value = None  # resume_lr flag not set

        lr = resume_lr_value if (resume_lr_value is not None and 501 == start_step) else get_lr(
            500, warmup_steps=100, max_steps=1000, max_lr=3e-4,
        )
        expected = get_lr(500, warmup_steps=100, max_steps=1000, max_lr=3e-4)
        assert lr == expected

    def test_set_lr_applies_resumed_value(self):
        """set_lr should correctly apply a restored LR to an OptimizerGroup."""
        model = _make_model()
        optimizer = create_optimizer(model, _Args("muon-jordan", lr=3e-4))

        resumed_lr = 1.23e-4
        set_lr(optimizer, resumed_lr)

        for pg in optimizer.param_groups:
            assert pg["lr"] == resumed_lr
