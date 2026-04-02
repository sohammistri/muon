"""
Unified evaluation script for NMT checkpoints.

Evaluates translation quality on up to three datasets:
  - samanantar val split (teacher-forcing loss + generation metrics)
  - FLORES+ devtest (generation metrics only)
  - IITB en-hi test (generation metrics only)

Metrics: BLEU, chrF++, and optionally COMET.

Examples:

    # Evaluate all checkpoints (BLEU + chrF++ only, no COMET)
    python -m attention_NMT.eval --checkpoint-dir attention_NMT/checkpoints --no-comet

    # Evaluate a specific step on FLORES+ only
    python -m attention_NMT.eval --checkpoint-dir attention_NMT/checkpoints --step 5000 --eval flores

    # Evaluate on IITB test set only
    python -m attention_NMT.eval --checkpoint-dir attention_NMT/checkpoints --step 5000 --eval iitb

    # Full evaluation with COMET (requires unbabel-comet installed separately)
    python -m attention_NMT.eval --checkpoint-dir attention_NMT/checkpoints --eval val,flores,iitb
"""

import os
import sys
import csv
import math
import time
import argparse

import torch
import torch.nn.functional as F
import sacrebleu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb
from GPT2.common import (
    compute_init, compute_cleanup, print0,
    get_base_dir, autodetect_device_type, DummyWandb, LocalMetricLogger,
)
from GPT2.checkpoint import load_checkpoint, find_all_steps
from attention_NMT.model import Transformer
from attention_NMT.tokenizer import get_tokenizer
from attention_NMT.data import get_nmt_loaders
from attention_NMT.eval_dataset import load_flores_eval
from attention_NMT.eval_dataset_iitb import load_iitb_eval


# -----------------------------------------------------------------------------
# Model loading

def load_nmt_model(checkpoint_dir, step, device):
    """Load an NMT Transformer from a training checkpoint.

    Returns:
        (model, context_window, step)
    """
    ckpt = load_checkpoint(checkpoint_dir, step, device)
    meta_args = ckpt["meta"]["args"]
    state_dict = ckpt["model_state_dict"]

    # Embedding shape is (vocab_size + 1, emb_dim) because pad_id = vocab_size
    num_embeddings = state_dict["backbone.token_emb.weight"].shape[0]
    vocab_size = num_embeddings - 1
    pad_id = vocab_size

    num_heads = meta_args.get("num_heads") or meta_args["emb_dim"] // 64
    model = Transformer(
        vocab_size=vocab_size,
        emb_dim=meta_args["emb_dim"],
        num_heads=num_heads,
        depth=meta_args["depth"],
        context_window=meta_args["context_window"],
        dropout=meta_args.get("dropout", 0.1),
        pad_id=pad_id,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, meta_args["context_window"], step


# -----------------------------------------------------------------------------
# Greedy decoding

@torch.no_grad()
def greedy_decode(model, src_ids, src_pad_mask, bos_id, eos_id, pad_id,
                  max_len, precision_ctx):
    """
    Greedy autoregressive decoding with encoder output caching.

    Args:
        model: raw Transformer in eval mode
        src_ids: (B, T_src) source token IDs
        src_pad_mask: (B, T_src) bool mask, True = real token
        bos_id, eos_id, pad_id: special token IDs
        max_len: maximum target sequence length to generate
        precision_ctx: autocast context manager

    Returns:
        list[list[int]]: decoded token IDs per batch element (BOS excluded, up to EOS exclusive)
    """
    B, T_src = src_ids.shape
    device = src_ids.device

    # --- Encode source (run once, cache output) ---
    with precision_ctx:
        src_emb = model.backbone.drop(
            model.backbone.token_emb(src_ids) + model.backbone.pos_emb[:, :T_src, :]
        )
        src_mask = None
        if src_pad_mask is not None:
            src_mask = torch.where(
                src_pad_mask[:, None, None, :], 0.0, float('-inf')
            )  # (B, 1, 1, T_src)

        enc_out = src_emb
        for block in model.backbone.encoder_blocks:
            enc_out = block(enc_out, src_mask=src_mask)
        enc_out = model.backbone.enc_ln_f(enc_out)

    # --- Decode autoregressively ---
    tgt_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        with precision_ctx:
            T_tgt = tgt_ids.shape[1]
            tgt_emb = model.backbone.drop(
                model.backbone.token_emb(tgt_ids)
                + model.backbone.pos_emb[:, :T_tgt, :]
            )
            causal_mask = torch.triu(
                torch.full((T_tgt, T_tgt), float('-inf'), device=device),
                diagonal=1,
            )[None, None, :, :]  # (1, 1, T_tgt, T_tgt)

            dec_out = tgt_emb
            for block in model.backbone.decoder_blocks:
                dec_out = block(dec_out, enc_out,
                                causal_pad_mask=causal_mask, cross_mask=src_mask)
            dec_out = model.backbone.dec_ln_f(dec_out)
            logits = model.head(dec_out)  # (B, T_tgt, V)

        next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
        next_token = next_token.masked_fill(done, pad_id)
        tgt_ids = torch.cat([tgt_ids, next_token.unsqueeze(1)], dim=1)
        done = done | (next_token == eos_id)
        if done.all():
            break

    # Extract decoded sequences (skip BOS, stop at EOS)
    results = []
    for i in range(B):
        tokens = tgt_ids[i, 1:].tolist()  # skip BOS
        if eos_id in tokens:
            tokens = tokens[:tokens.index(eos_id)]
        results.append(tokens)

    return results


# -----------------------------------------------------------------------------
# Helper: decode token IDs to text, stripping special tokens

def ids_to_text(token_ids, tokenizer, bos_id, eos_id, pad_id):
    """Strip BOS/EOS/PAD from a 1-D token ID list, then decode to string."""
    filtered = [t for t in token_ids if t not in (bos_id, eos_id, pad_id)]
    return tokenizer.decode(filtered)


def batch_ids_to_texts(batch_tensor, tokenizer, bos_id, eos_id, pad_id):
    """Convert a (B, T) tensor of token IDs to a list of strings."""
    texts = []
    for i in range(batch_tensor.shape[0]):
        tokens = batch_tensor[i].tolist()
        texts.append(ids_to_text(tokens, tokenizer, bos_id, eos_id, pad_id))
    return texts


# -----------------------------------------------------------------------------
# Metric computation

def compute_translation_metrics(hypotheses, references, sources=None,
                                comet_model=None):
    """
    Compute BLEU, chrF++, and optionally COMET.

    Args:
        hypotheses: list[str] of model translations
        references: list[str] of reference translations
        sources: list[str] of source sentences (needed for COMET)
        comet_model: loaded COMET model or None

    Returns:
        dict with "bleu", "chrf", and optionally "comet"
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    results = {
        "bleu": bleu.score,
        "chrf": chrf.score,
    }

    if comet_model is not None and sources is not None:
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        comet_output = comet_model.predict(data, batch_size=64, gpus=0)
        results["comet"] = comet_output.system_score

    return results


# -----------------------------------------------------------------------------
# Teacher-forcing loss evaluation

@torch.no_grad()
def evaluate_loss(model, val_loader, val_steps, pad_id, precision_ctx):
    """
    Compute teacher-forcing cross-entropy loss and perplexity on val data.

    Returns:
        dict with "loss" and "perplexity"
    """
    model.eval()
    total_loss = 0.0

    for _ in range(val_steps):
        src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = next(val_loader)
        with precision_ctx:
            logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_label.view(-1),
            ignore_index=pad_id,
        )
        total_loss += loss.item()

    avg_loss = total_loss / val_steps
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


# -----------------------------------------------------------------------------
# Val generation evaluation

@torch.no_grad()
def evaluate_val_generation(model, val_loader, val_gen_steps, tokenizer,
                            bos_id, eos_id, pad_id, max_len, precision_ctx,
                            device, comet_model=None):
    """
    Run greedy decoding on samanantar val batches and compute translation metrics.

    Returns:
        dict with "bleu", "chrf", and optionally "comet"
    """
    all_hypotheses = []
    all_references = []
    all_sources = []

    for _ in range(val_gen_steps):
        src_ids, tgt_input, tgt_label, src_pad_mask, tgt_pad_mask = next(val_loader)

        decoded_ids = greedy_decode(
            model, src_ids, src_pad_mask,
            bos_id, eos_id, pad_id, max_len, precision_ctx,
        )

        # Decode hypotheses
        hyps = [tokenizer.decode(ids) for ids in decoded_ids]
        all_hypotheses.extend(hyps)

        # Decode references from tgt_label (tokens..., EOS, PAD...)
        refs = batch_ids_to_texts(tgt_label, tokenizer, bos_id, eos_id, pad_id)
        all_references.extend(refs)

        # Decode sources for COMET
        srcs = batch_ids_to_texts(src_ids, tokenizer, bos_id, eos_id, pad_id)
        all_sources.extend(srcs)

    return compute_translation_metrics(
        all_hypotheses, all_references, all_sources, comet_model,
    )


# -----------------------------------------------------------------------------
# FLORES+ evaluation

@torch.no_grad()
def evaluate_flores(model, tokenizer, bos_id, eos_id, pad_id, max_len,
                    batch_size, precision_ctx, device, comet_model=None):
    """
    Evaluate on FLORES+ devtest en-hi pairs.

    Returns:
        dict with "bleu", "chrf", and optionally "comet"
    """
    pairs = load_flores_eval()
    src_texts = [p["src"] for p in pairs]
    ref_texts = [p["tgt"] for p in pairs]

    # Tokenize all sources
    src_encoded = tokenizer.encode(
        src_texts,
        prepend="<|bos|>",
        append="<|eos|>",
    )

    all_hypotheses = []
    num_batches = (len(src_encoded) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(src_encoded))
        batch_tokens = src_encoded[start:end]

        # Pad to batch max length
        max_src_len = max(len(s) for s in batch_tokens)
        padded = [s + [pad_id] * (max_src_len - len(s)) for s in batch_tokens]

        src_ids = torch.tensor(padded, dtype=torch.long, device=device)
        src_pad_mask = (src_ids != pad_id)

        decoded_ids = greedy_decode(
            model, src_ids, src_pad_mask,
            bos_id, eos_id, pad_id, max_len, precision_ctx,
        )
        hyps = [tokenizer.decode(ids) for ids in decoded_ids]
        all_hypotheses.extend(hyps)

    # Print samples
    print0("\n--- FLORES+ Samples ---")
    for i in range(min(3, len(src_texts))):
        print0(f"  SRC: {src_texts[i][:120]}...")
        print0(f"  REF: {ref_texts[i][:120]}...")
        print0(f"  HYP: {all_hypotheses[i][:120]}...")
        print0()

    return compute_translation_metrics(
        all_hypotheses, ref_texts, src_texts, comet_model,
    )


# -----------------------------------------------------------------------------
# IITB evaluation

@torch.no_grad()
def evaluate_iitb(model, tokenizer, bos_id, eos_id, pad_id, max_len,
                  batch_size, precision_ctx, device, comet_model=None):
    """
    Evaluate on IITB en-hi test pairs.

    Returns:
        dict with "bleu", "chrf", and optionally "comet"
    """
    pairs = load_iitb_eval()
    src_texts = [p["src"] for p in pairs]
    ref_texts = [p["tgt"] for p in pairs]

    # Tokenize all sources
    src_encoded = tokenizer.encode(
        src_texts,
        prepend="<|bos|>",
        append="<|eos|>",
    )

    all_hypotheses = []
    num_batches = (len(src_encoded) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(src_encoded))
        batch_tokens = src_encoded[start:end]

        # Pad to batch max length
        max_src_len = max(len(s) for s in batch_tokens)
        padded = [s + [pad_id] * (max_src_len - len(s)) for s in batch_tokens]

        src_ids = torch.tensor(padded, dtype=torch.long, device=device)
        src_pad_mask = (src_ids != pad_id)

        decoded_ids = greedy_decode(
            model, src_ids, src_pad_mask,
            bos_id, eos_id, pad_id, max_len, precision_ctx,
        )
        hyps = [tokenizer.decode(ids) for ids in decoded_ids]
        all_hypotheses.extend(hyps)

    # Print samples
    print0("\n--- IITB Test Samples ---")
    for i in range(min(3, len(src_texts))):
        print0(f"  SRC: {src_texts[i][:120]}...")
        print0(f"  REF: {ref_texts[i][:120]}...")
        print0(f"  HYP: {all_hypotheses[i][:120]}...")
        print0()

    return compute_translation_metrics(
        all_hypotheses, ref_texts, src_texts, comet_model,
    )


# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="NMT checkpoint evaluation")
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing NMT checkpoints')
    parser.add_argument('--step', type=int, default=None,
                        help='Evaluate specific step (default: all)')
    parser.add_argument('--eval', type=str, default='val,flores',
                        help='Comma-separated: val,flores,iitb (default: val,flores)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for generation (default: 32)')
    parser.add_argument('--val-steps', type=int, default=20,
                        help='Number of val batches for loss eval (default: 20)')
    parser.add_argument('--val-gen-steps', type=int, default=10,
                        help='Number of val batches for generation eval (default: 10)')
    parser.add_argument('--max-decode-len', type=int, default=256,
                        help='Max target length for greedy decoding (default: 256)')
    parser.add_argument('--segment', type=str, default='hi',
                        help='Language segment (default: hi)')
    parser.add_argument('--device-type', type=str, default='',
                        help='cuda|cpu|mps (empty = autodetect)')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16',
                        help='Inference precision (default: bf16)')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction,
                        default=True, help='Enable W&B logging (--no-wandb to disable)')
    parser.add_argument('--no-comet', action='store_true',
                        help='Skip COMET evaluation (avoids model download)')
    args = parser.parse_args()

    # Parse eval modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'val', 'flores', 'iitb'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Device / DDP setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # Load tokenizer
    tokenizer = get_tokenizer(args.segment)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = vocab_size
    bos_id = tokenizer.get_bos_token_id()
    eos_id = tokenizer.get_eos_token_id()

    # Discover checkpoint steps
    if args.step is not None:
        steps_to_eval = [args.step]
    else:
        steps_to_eval = find_all_steps(args.checkpoint_dir)
        if not steps_to_eval:
            print0(f"No checkpoints found in {args.checkpoint_dir}")
            compute_cleanup()
            return

    print0(f"Checkpoint directory: {args.checkpoint_dir}")
    print0(f"Steps to evaluate: {steps_to_eval}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Precision context
    if args.precision == 'bf16' and device_type in ('cuda', 'mps'):
        precision_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        precision_ctx = torch.amp.autocast(device_type=device_type, enabled=False)

    # Init wandb
    run_name = f"nmt-eval-{os.path.basename(args.checkpoint_dir)}"
    eval_config = {
        "checkpoint_dir": args.checkpoint_dir,
        "eval_modes": sorted(eval_modes),
        "batch_size": args.batch_size,
        "max_decode_len": args.max_decode_len,
        "precision": args.precision,
    }
    eval_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_logs")
    if args.wandb and ddp_rank == 0:
        wandb.init(
            project="muon",
            name=run_name,
            config=eval_config,
        )
        wb = wandb
    elif not args.wandb and ddp_rank == 0:
        wb = LocalMetricLogger(project="muon", name=run_name, config=eval_config,
                               log_dir=eval_log_dir)
    else:
        wb = DummyWandb()

    # Load COMET model (once, reused across checkpoints)
    comet_model = None
    if not args.no_comet:
        try:
            from comet import download_model, load_from_checkpoint
            print0("Loading COMET model (Unbabel/wmt22-comet-da)...")
            model_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(model_path)
            print0("COMET model loaded.")
        except ImportError:
            print0("WARNING: unbabel-comet not installed. Skipping COMET. "
                   "Install with: pip install unbabel-comet (requires transformers<5)")
        except Exception as e:
            print0(f"WARNING: Failed to load COMET model: {e}. Skipping COMET.")

    # Aggregate results
    aggregate_results = []

    for step in steps_to_eval:
        print0("\n" + "#" * 80)
        print0(f"Evaluating checkpoint at step {step}")
        print0("#" * 80)

        # Load model
        model, context_window, loaded_step = load_nmt_model(
            args.checkpoint_dir, step, device)
        print0(f"Loaded model (step={loaded_step}, context_window={context_window})")

        step_results = {"step": step}
        log_dict = {}

        # --- Val split evaluation ---
        if 'val' in eval_modes:
            print0("\n" + "=" * 60)
            print0("Samanantar Val Evaluation")
            print0("=" * 60)

            _, val_loader, _, _ = get_nmt_loaders(
                B=args.batch_size, T=context_window,
                segment=args.segment, device=device,
            )

            # Teacher-forcing loss
            print0("Computing loss/perplexity...")
            loss_results = evaluate_loss(
                model, val_loader, args.val_steps, pad_id, precision_ctx)
            print0(f"  loss: {loss_results['loss']:.4f}  "
                   f"perplexity: {loss_results['perplexity']:.4f}")

            step_results["val_loss"] = loss_results["loss"]
            step_results["val_ppl"] = loss_results["perplexity"]
            log_dict["eval/val_loss"] = loss_results["loss"]
            log_dict["eval/val_perplexity"] = loss_results["perplexity"]

            # Generation metrics
            print0("Running greedy decoding on val split...")
            t0 = time.time()
            val_gen = evaluate_val_generation(
                model, val_loader, args.val_gen_steps, tokenizer,
                bos_id, eos_id, pad_id, args.max_decode_len,
                precision_ctx, device, comet_model,
            )
            elapsed = time.time() - t0
            print0(f"  BLEU: {val_gen['bleu']:.2f}  "
                   f"chrF++: {val_gen['chrf']:.2f}  "
                   f"COMET: {val_gen.get('comet', 'N/A')}  "
                   f"({elapsed:.1f}s)")

            step_results["val_bleu"] = val_gen["bleu"]
            step_results["val_chrf"] = val_gen["chrf"]
            log_dict["eval/val_bleu"] = val_gen["bleu"]
            log_dict["eval/val_chrf"] = val_gen["chrf"]
            if "comet" in val_gen:
                step_results["val_comet"] = val_gen["comet"]
                log_dict["eval/val_comet"] = val_gen["comet"]

        # --- FLORES+ evaluation ---
        if 'flores' in eval_modes:
            print0("\n" + "=" * 60)
            print0("FLORES+ Devtest Evaluation")
            print0("=" * 60)

            t0 = time.time()
            flores_results = evaluate_flores(
                model, tokenizer, bos_id, eos_id, pad_id,
                args.max_decode_len, args.batch_size, precision_ctx,
                device, comet_model,
            )
            elapsed = time.time() - t0
            print0(f"  BLEU: {flores_results['bleu']:.2f}  "
                   f"chrF++: {flores_results['chrf']:.2f}  "
                   f"COMET: {flores_results.get('comet', 'N/A')}  "
                   f"({elapsed:.1f}s)")

            step_results["flores_bleu"] = flores_results["bleu"]
            step_results["flores_chrf"] = flores_results["chrf"]
            log_dict["eval/flores_bleu"] = flores_results["bleu"]
            log_dict["eval/flores_chrf"] = flores_results["chrf"]
            if "comet" in flores_results:
                step_results["flores_comet"] = flores_results["comet"]
                log_dict["eval/flores_comet"] = flores_results["comet"]

        # --- IITB evaluation ---
        if 'iitb' in eval_modes:
            print0("\n" + "=" * 60)
            print0("IITB Test Evaluation")
            print0("=" * 60)

            t0 = time.time()
            iitb_results = evaluate_iitb(
                model, tokenizer, bos_id, eos_id, pad_id,
                args.max_decode_len, args.batch_size, precision_ctx,
                device, comet_model,
            )
            elapsed = time.time() - t0
            print0(f"  BLEU: {iitb_results['bleu']:.2f}  "
                   f"chrF++: {iitb_results['chrf']:.2f}  "
                   f"COMET: {iitb_results.get('comet', 'N/A')}  "
                   f"({elapsed:.1f}s)")

            step_results["iitb_bleu"] = iitb_results["bleu"]
            step_results["iitb_chrf"] = iitb_results["chrf"]
            log_dict["eval/iitb_bleu"] = iitb_results["bleu"]
            log_dict["eval/iitb_chrf"] = iitb_results["chrf"]
            if "comet" in iitb_results:
                step_results["iitb_comet"] = iitb_results["comet"]
                log_dict["eval/iitb_comet"] = iitb_results["comet"]

        # Log to wandb
        if log_dict:
            wb.log(log_dict, step=step)

        aggregate_results.append(step_results)

        # Write per-step CSV
        if ddp_rank == 0:
            base_dir = get_base_dir()
            eval_dir = os.path.join(base_dir, "nmt_eval")
            os.makedirs(eval_dir, exist_ok=True)

            csv_path = os.path.join(eval_dir, f"nmt_step_{step:06d}.csv")
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "samanantar_val", "flores_devtest", "iitb_test"])

                def _val(key):
                    return f"{step_results[key]:.4f}" if key in step_results else ""

                writer.writerow(["loss", _val("val_loss"), "", ""])
                writer.writerow(["perplexity", _val("val_ppl"), "", ""])
                writer.writerow(["bleu", _val("val_bleu"), _val("flores_bleu"), _val("iitb_bleu")])
                writer.writerow(["chrf", _val("val_chrf"), _val("flores_chrf"), _val("iitb_chrf")])
                writer.writerow(["comet", _val("val_comet"), _val("flores_comet"), _val("iitb_comet")])

            print0(f"Results written to: {csv_path}")

        # Free memory
        del model
        if device_type == "cuda":
            torch.cuda.empty_cache()

    # Write aggregate CSV
    if ddp_rank == 0 and aggregate_results:
        base_dir = get_base_dir()
        eval_dir = os.path.join(base_dir, "nmt_eval")
        os.makedirs(eval_dir, exist_ok=True)

        dirname = os.path.basename(args.checkpoint_dir)
        agg_path = os.path.join(eval_dir, f"nmt_aggregate_{dirname}.csv")

        all_keys = []
        for r in aggregate_results:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)

        with open(agg_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(aggregate_results)

        print0(f"\nAggregate results written to: {agg_path}")

    wb.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
