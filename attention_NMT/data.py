"""
Data loading for NMT training.

Provides a clean API analogous to GPT2/data.py, wrapping the distributed
tokenizing dataloaders from attention_NMT/dataloader.py.

These are infinite generators. Data is already transferred to device
inside the generator.

Usage:
    train_loader, val_loader, vocab_size, pad_id = get_nmt_loaders(B=64, T=256)

    # Training (step-based)
    for step in range(max_steps):
        src, tgt_in, tgt_lbl, src_mask, tgt_mask, data_state = next(train_loader)

    # Validation
    for val_step in range(val_steps):
        src, tgt_in, tgt_lbl, src_mask, tgt_mask = next(val_loader)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attention_NMT.tokenizer import get_tokenizer
from attention_NMT.dataloader import (
    nmt_distributed_data_loader_with_state,
    nmt_distributed_data_loader,
)


def get_nmt_loaders(B, T, segment="hi", device="cuda", resume_state_dict=None):
    """
    NMT dataloaders.

    Args:
        B: batch size (number of sentence pairs per batch)
        T: max sequence length (content tokens; BOS/EOS added internally)
        segment: language pair segment (default: "hi" for en-hi)
        device: target device for tensors (default: "cuda")
        resume_state_dict: optional dict with {"pq_idx", "rg_idx", "epoch"}
                           for resuming the train loader from a checkpoint

    Returns:
        train_loader: infinite generator yielding
            (src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask, state_dict)
        val_loader: infinite generator yielding
            (src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask)
        vocab_size: int, vocabulary size (actual tokens, not including PAD)
        pad_id: int, the PAD token ID (= vocab_size)
    """
    tokenizer = get_tokenizer(segment)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = vocab_size  # PAD is one past last valid token

    train_loader = nmt_distributed_data_loader_with_state(
        tokenizer, B, T, segment, split="train",
        device=device, resume_state_dict=resume_state_dict,
    )
    val_loader = nmt_distributed_data_loader(
        tokenizer, B, T, segment, split="val",
        device=device,
    )

    return train_loader, val_loader, vocab_size, pad_id
