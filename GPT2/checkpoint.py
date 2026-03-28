"""
Checkpoint save/load utilities for GPT-2 training.

File naming convention (flat layout in checkpoint_dir):
    model_{step:06d}.pt          — model state_dict (rank 0 only)
    optim_{step:06d}_rank{rank}.pt — optimizer state_dict (per rank, sharded in DDP)
    meta_{step:06d}.json         — metadata as JSON (rank 0 only)
"""

import glob
import json
import logging
import os
import re

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


def save_checkpoint(checkpoint_dir, step, model_state, optimizer_state,
                    meta_data, rank=0, use_ddp=False):
    """Save a training checkpoint.

    Args:
        checkpoint_dir: directory to save checkpoint files
        step: current training step
        model_state: model.state_dict()
        optimizer_state: optimizer.state_dict() (list for OptimizerGroup)
        meta_data: dict with step, val_loss, args, data_state, lr_state
        rank: DDP rank (0 for single-GPU)
        use_ddp: whether DDP is active
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if rank == 0:
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_state, model_path)
        log.info(f"Saved model parameters to: {model_path}")

        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        log.info(f"Saved metadata to: {meta_path}")

    # Optimizer state is sharded across ranks — each rank saves its own
    optim_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
    torch.save(optimizer_state, optim_path)
    log.info(f"Saved optimizer state to: {optim_path}")

    if use_ddp:
        dist.barrier()


def load_checkpoint(checkpoint_dir, step, device, rank=0, use_ddp=False):
    """Load a training checkpoint.

    Args:
        checkpoint_dir: directory containing checkpoint files
        step: step number to load
        device: device to map tensors to
        rank: DDP rank (0 for single-GPU)
        use_ddp: whether DDP is active

    Returns:
        dict with keys: step, model_state_dict, optimizer_state_dict, meta
    """
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_state = torch.load(model_path, map_location=device, weights_only=True)

    optim_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
    optimizer_state = torch.load(optim_path, map_location=device, weights_only=True)

    if use_ddp:
        dist.barrier()

    return {
        "step": meta["step"],
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "meta": meta,
    }


def find_all_steps(checkpoint_dir):
    """Find all checkpoint steps in a directory, sorted ascending.

    Returns:
        List of step numbers found, sorted ascending. Empty list if none.
    """
    if not os.path.isdir(checkpoint_dir):
        return []

    pattern = os.path.join(checkpoint_dir, "meta_*.json")
    meta_files = glob.glob(pattern)
    if not meta_files:
        return []

    steps = []
    for path in meta_files:
        filename = os.path.basename(path)
        match = re.match(r"meta_(\d+)\.json$", filename)
        if match:
            steps.append(int(match.group(1)))

    return sorted(steps)


def find_latest_step(checkpoint_dir):
    """Find the latest checkpoint step in a directory.

    Returns:
        The highest step number found, or None if no checkpoints exist.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, "meta_*.json")
    meta_files = glob.glob(pattern)
    if not meta_files:
        return None

    steps = []
    for path in meta_files:
        filename = os.path.basename(path)
        match = re.match(r"meta_(\d+)\.json$", filename)
        if match:
            steps.append(int(match.group(1)))

    return max(steps) if steps else None
