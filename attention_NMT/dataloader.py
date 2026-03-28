"""
Distributed dataloaders for NMT (Neural Machine Translation).

Reads parallel src-tgt sentence pairs from parquet files, tokenizes with
a shared BPE tokenizer, and produces dynamically-padded batches with
attention masks suitable for encoder-decoder models.

Key properties:
- Every src sequence: [BOS, src_tokens..., EOS, PAD...]
- Every tgt split into decoder input [BOS, tgt_tokens..., PAD...]
  and labels [tgt_tokens..., EOS, PAD...]
- Dynamic padding: each batch padded to its longest sequence (not global max)
- Length bucketing: sentences sorted by length within pools to minimize padding
- DDP-aware: row groups sharded across ranks
- Resumable: tracks (pq_idx, rg_idx, epoch) for checkpoint resume
"""

import torch
import pyarrow.parquet as pq
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from GPT2.common import get_dist_info
from attention_NMT.dataset import list_parquet_files


def _parallel_sentence_batches(segment, split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over parallel sentence batches from parquet files.

    Handles DDP sharding and approximate resume. Each yield is
    (src_texts, tgt_texts, (pq_idx, rg_idx, epoch)) where src_texts and
    tgt_texts are lists of strings.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files(segment=segment)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                src_texts = rg.column('src').to_pylist()
                tgt_texts = rg.column('tgt').to_pylist()
                for i in range(0, len(src_texts), tokenizer_batch_size):
                    yield (
                        src_texts[i:i + tokenizer_batch_size],
                        tgt_texts[i:i + tokenizer_batch_size],
                        (pq_idx, rg_idx, epoch),
                    )
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def nmt_distributed_data_loader_with_state(
    tokenizer, B, T, segment, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    bucket_size=None,
):
    """
    NMT dataloader with dynamic padding and length bucketing.

    Args:
        tokenizer: NMT tokenizer (shared for src and tgt)
        B: batch size (number of sentence pairs per batch)
        T: max sequence length (tokens, excluding BOS/EOS which are added internally)
        segment: language segment (e.g. "hi")
        split: "train" or "val"
        tokenizer_threads: threads for parallel tokenization
        tokenizer_batch_size: how many sentences to pull per parquet read
        device: target device
        resume_state_dict: optional dict with {"pq_idx", "rg_idx", "epoch"}
        bucket_size: pool size for length bucketing (default: B * 100)

    Yields:
        src_ids:       (B, T_src_batch) LongTensor - encoder input [BOS, src..., EOS, PAD...]
        tgt_input_ids: (B, T_tgt_batch) LongTensor - decoder input [BOS, tgt..., PAD...]
        tgt_label_ids: (B, T_tgt_batch) LongTensor - decoder labels [tgt..., EOS, PAD...]
        src_pad_mask:  (B, T_src_batch) BoolTensor  - True where NOT padded
        tgt_pad_mask:  (B, T_tgt_batch) BoolTensor  - True where NOT padded
        state_dict:    dict with {"pq_idx", "rg_idx", "epoch"}
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    if bucket_size is None:
        bucket_size = B * 100

    bos_id = tokenizer.get_bos_token_id()
    eos_id = tokenizer.get_eos_token_id()
    pad_id = tokenizer.get_vocab_size()  # PAD = one past last valid token
    max_tok_len = T + 2  # T content tokens + BOS + EOS

    batches = _parallel_sentence_batches(segment, split, resume_state_dict, tokenizer_batch_size)
    use_cuda = device == "cuda"

    sentence_pool = []  # list of (src_token_ids, tgt_token_ids) tuples
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_pool():
        nonlocal pq_idx, rg_idx, epoch
        src_texts, tgt_texts, (pq_idx, rg_idx, epoch) = next(batches)
        src_token_lists = tokenizer.encode(
            src_texts, prepend=bos_id, append=eos_id, num_threads=tokenizer_threads,
        )
        tgt_token_lists = tokenizer.encode(
            tgt_texts, prepend=bos_id, append=eos_id, num_threads=tokenizer_threads,
        )
        for src_toks, tgt_toks in zip(src_token_lists, tgt_token_lists):
            # Truncate to max_tok_len, preserving BOS at start and forcing EOS at end
            if len(src_toks) > max_tok_len:
                src_toks = src_toks[:max_tok_len - 1] + [eos_id]
            if len(tgt_toks) > max_tok_len:
                tgt_toks = tgt_toks[:max_tok_len - 1] + [eos_id]
            sentence_pool.append((src_toks, tgt_toks))

    while True:
        # Fill pool to at least bucket_size
        while len(sentence_pool) < bucket_size:
            refill_pool()

        # Sort by max length of src/tgt for efficient padding
        sentence_pool.sort(key=lambda pair: max(len(pair[0]), len(pair[1])))

        # Chunk into batches of B, keep remainder for next round
        num_full_batches = len(sentence_pool) // B
        batch_list = []
        for bi in range(num_full_batches):
            batch_list.append(sentence_pool[bi * B : (bi + 1) * B])
        sentence_pool = sentence_pool[num_full_batches * B :]

        for batch in batch_list:
            max_src_len = max(len(s) for s, t in batch)
            max_tgt_len = max(len(t) for s, t in batch)
            tgt_seq_len = max_tgt_len - 1  # decoder input/label length

            # Allocate padded tensors on pinned CPU memory
            src_ids = torch.full((B, max_src_len), pad_id, dtype=torch.long, pin_memory=use_cuda)
            tgt_input = torch.full((B, tgt_seq_len), pad_id, dtype=torch.long, pin_memory=use_cuda)
            tgt_label = torch.full((B, tgt_seq_len), pad_id, dtype=torch.long, pin_memory=use_cuda)

            for i, (src_toks, tgt_toks) in enumerate(batch):
                src_ids[i, :len(src_toks)] = torch.tensor(src_toks, dtype=torch.long)
                tgt_len = len(tgt_toks) - 1
                tgt_input[i, :tgt_len] = torch.tensor(tgt_toks[:-1], dtype=torch.long)
                tgt_label[i, :tgt_len] = torch.tensor(tgt_toks[1:], dtype=torch.long)

            # Masks: True where token is real (not padding)
            src_pad_mask = (src_ids != pad_id)
            tgt_pad_mask = (tgt_label != pad_id)

            # Transfer to device
            src_ids = src_ids.to(device, non_blocking=use_cuda)
            tgt_input = tgt_input.to(device, non_blocking=use_cuda)
            tgt_label = tgt_label.to(device, non_blocking=use_cuda)
            src_pad_mask = src_pad_mask.to(device, non_blocking=use_cuda)
            tgt_pad_mask = tgt_pad_mask.to(device, non_blocking=use_cuda)

            state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
            yield src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask, state_dict


def nmt_distributed_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask, state_dict in \
            nmt_distributed_data_loader_with_state(*args, **kwargs):
        yield src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask
