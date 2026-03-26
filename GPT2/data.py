"""
Data loading for GPT-2 pretraining.

Provides a clean API analogous to CNN/data.py and MLP/data.py,
wrapping the distributed tokenizing dataloaders from GPT2/dataloader.py.

Unlike CNN/MLP loaders (which return finite torch DataLoaders), these are
infinite generators. Data is already transferred to device inside the
generator (single HtoD copy).

Usage:
    train_loader, val_loader, vocab_size = get_pretraining_loaders(B=64, T=2048)

    # Training (step-based, not epoch-based)
    for step in range(max_steps):
        inputs, targets, data_state = next(train_loader)
        # inputs: (B, T) LongTensor on device
        # targets: (B, T) LongTensor on device
        # data_state: dict for checkpoint resume

    # Validation
    for val_step in range(val_steps):
        inputs, targets = next(val_loader)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from GPT2.tokenizer import get_tokenizer
from GPT2.dataloader import (
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
    tokenizing_distributed_data_loader_bos_bestfit,
)


def get_pretraining_loaders(B, T, device="cuda", resume_state_dict=None):
    """
    GPT-2 pretraining dataloaders.

    Args:
        B: batch size (number of sequences per batch)
        T: sequence length (context window size)
        device: target device for tensors (default: "cuda")
        resume_state_dict: optional dict with {"pq_idx", "rg_idx", "epoch"}
                           for resuming the train loader from a checkpoint

    Returns:
        train_loader: infinite generator yielding (inputs, targets, state_dict)
        val_loader: infinite generator yielding (inputs, targets)
        vocab_size: int, vocabulary size for model construction
    """
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, B, T, split="train",
        device=device, resume_state_dict=resume_state_dict,
    )
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B, T, split="val",
        device=device,
    )

    return train_loader, val_loader, vocab_size
