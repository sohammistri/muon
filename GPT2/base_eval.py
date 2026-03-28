"""
Unified evaluation script for GPT-2 checkpoints.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model (HF models only)

Default is core,bpb: --eval core,bpb

Examples:

    # Evaluate all checkpoints in a directory
    python GPT2/base_eval.py --checkpoint-dir GPT2/checkpoints --eval core,bpb

    # Evaluate a specific checkpoint step
    python GPT2/base_eval.py --checkpoint-dir GPT2/checkpoints --step 5000 --eval core

    # Evaluate a HuggingFace model (e.g. GPT-2 124M)
    python GPT2/base_eval.py --hf-path openai-community/gpt2

    # Multi-GPU evaluation
    torchrun --nproc_per_node=8 GPT2/base_eval.py --checkpoint-dir GPT2/checkpoints
"""
import os
import sys
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb
from GPT2.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock, DummyWandb, LocalMetricLogger
from GPT2.tokenizer import HuggingFaceTokenizer, get_token_bytes, get_tokenizer
from GPT2.checkpoint import load_checkpoint, find_all_steps
from GPT2.core_eval import evaluate_task
from GPT2.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from GPT2.loss_eval import evaluate_bpb
from GPT2.model import GPT2

# -----------------------------------------------------------------------------
# Model wrappers

class GPT2ModelWrapper:
    """Wraps our GPT2 model to match the interface expected by core_eval and loss_eval."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


class HFModelWrapper:
    """Lightweight wrapper to give HuggingFace models a compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


# -----------------------------------------------------------------------------
# Loading utilities

def load_gpt2_model(checkpoint_dir, step, device):
    """Load a GPT2 model from a training checkpoint.

    Returns:
        (wrapped_model, sequence_len, step)
    """
    ckpt = load_checkpoint(checkpoint_dir, step, device)
    meta_args = ckpt["meta"]["args"]
    state_dict = ckpt["model_state_dict"]
    vocab_size = state_dict["backbone.token_emb.weight"].shape[0]
    model = GPT2(
        vocab_size=vocab_size,
        emb_dim=meta_args["emb_dim"],
        num_heads=meta_args["num_heads"],
        depth=meta_args["depth"],
        context_window=meta_args["context_window"],
        dropout=meta_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    seq_len = meta_args["context_window"]
    wrapped = GPT2ModelWrapper(model, max_seq_len=seq_len)
    return wrapped, seq_len, step


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = HFModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes


# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# Main


def evaluate_single_checkpoint(model, tokenizer, token_bytes, sequence_len,
                               eval_modes, args, ddp_rank, ddp_world_size, device):
    """Run all requested evaluations on a single loaded model.

    Returns:
        dict with core_results, bpb_results keys
    """
    core_results = None
    bpb_results = {}

    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        split_tokens = args.split_tokens
        if split_tokens % tokens_per_step != 0:
            split_tokens = (split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {split_tokens} (must be divisible by {tokens_per_step})")
        steps = split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)
        print0(f"CORE metric: {core_results['core_metric']:.4f}")

    return {"core_results": core_results, "bpb_results": bpb_results}


def main():
    parser = argparse.ArgumentParser(description="GPT-2 checkpoint evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb',
                        help='Comma-separated evaluations to run: core,bpb,sample (default: core,bpb)')
    parser.add_argument('--hf-path', type=str, default=None,
                        help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing training checkpoints')
    parser.add_argument('--step', type=int, default=None,
                        help='Evaluate specific step only (default: all steps)')
    parser.add_argument('--max-per-task', type=int, default=-1,
                        help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32,
                        help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288,
                        help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='',
                        help='cuda|cpu|mps (empty = autodetect)')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction,
                        default=True, help='Enable W&B logging (--no-wandb to disable)')
    args = parser.parse_args()

    # Validate args
    if args.hf_path is None and args.checkpoint_dir is None:
        parser.error("Must specify either --hf-path or --checkpoint-dir")

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # -------------------------------------------------------------------------
    # HuggingFace model path — single evaluation (original behavior)
    # -------------------------------------------------------------------------
    if args.hf_path is not None:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")

        print0(f"Evaluating model: {model_name}")
        print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

        if 'sample' in eval_modes:
            print0("\nSkipping sampling for HuggingFace models (not supported)")

        results = evaluate_single_checkpoint(
            model, tokenizer, token_bytes, sequence_len,
            eval_modes, args, ddp_rank, ddp_world_size, device)

        # Write CORE CSV
        if results["core_results"] and ddp_rank == 0:
            core_results = results["core_results"]
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")

        compute_cleanup()
        return

    # -------------------------------------------------------------------------
    # GPT2 checkpoint evaluation — iterate over all checkpoints
    # -------------------------------------------------------------------------
    checkpoint_dir = args.checkpoint_dir

    # Discover checkpoint steps
    if args.step is not None:
        steps_to_eval = [args.step]
    else:
        steps_to_eval = find_all_steps(checkpoint_dir)
        if not steps_to_eval:
            print0(f"No checkpoints found in {checkpoint_dir}")
            compute_cleanup()
            return

    print0(f"Checkpoint directory: {checkpoint_dir}")
    print0(f"Steps to evaluate: {steps_to_eval}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    if 'sample' in eval_modes:
        print0("\nSkipping sampling for GPT2 models (Engine not compatible)")
        eval_modes = eval_modes - {'sample'}

    # Load tokenizer once (shared across all checkpoints)
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    # Init wandb (or local file logger when W&B is unavailable)
    eval_config = {
        "checkpoint_dir": checkpoint_dir,
        "eval_modes": sorted(eval_modes),
        "max_per_task": args.max_per_task,
        "device_batch_size": args.device_batch_size,
        "split_tokens": args.split_tokens,
    }
    if args.wandb and ddp_rank == 0:
        wandb.init(
            project="muon",
            name=f"gpt2-eval-{os.path.basename(checkpoint_dir)}",
            config=eval_config,
        )
        wb = wandb
    elif ddp_rank == 0:
        wb = LocalMetricLogger(
            project="muon",
            name=f"gpt2-eval-{os.path.basename(checkpoint_dir)}",
            config=eval_config,
        )
    else:
        wb = DummyWandb()

    # Aggregate results across all steps
    aggregate_results = []

    for step in steps_to_eval:
        print0("\n" + "#"*80)
        print0(f"Evaluating checkpoint at step {step}")
        print0("#"*80)

        # Load model
        model, sequence_len, loaded_step = load_gpt2_model(checkpoint_dir, step, device)
        print0(f"Loaded model (step={loaded_step}, seq_len={sequence_len})")

        # Run evaluations
        results = evaluate_single_checkpoint(
            model, tokenizer, token_bytes, sequence_len,
            eval_modes, args, ddp_rank, ddp_world_size, device)

        core_results = results["core_results"]
        bpb_results = results["bpb_results"]

        # Build log dict for wandb
        log_dict = {}
        step_summary = {"step": step}

        if core_results:
            log_dict["eval/core_metric"] = core_results["core_metric"]
            step_summary["core_metric"] = core_results["core_metric"]
            for label, val in core_results["centered_results"].items():
                log_dict[f"eval/core/{label}"] = val
            step_summary.update(core_results["centered_results"])

        if bpb_results:
            for split_name, bpb in bpb_results.items():
                log_dict[f"eval/bpb/{split_name}"] = bpb
                step_summary[f"{split_name}_bpb"] = bpb

        # Log to wandb
        if log_dict:
            wb.log(log_dict, step=step)

        aggregate_results.append(step_summary)

        # Write per-step CORE CSV
        if core_results and ddp_rank == 0:
            base_dir = get_base_dir()
            ckpt_subdir = os.path.basename(checkpoint_dir.rstrip("/"))
            output_csv_path = os.path.join(base_dir, "base_eval", ckpt_subdir,
                                           f"gpt2_step_{step:06d}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"Results written to: {output_csv_path}")

        # Free memory before loading next checkpoint
        del model
        if device_type == "cuda":
            torch.cuda.empty_cache()

    # Write aggregate CSV
    if ddp_rank == 0 and aggregate_results:
        base_dir = get_base_dir()
        ckpt_subdir = os.path.basename(checkpoint_dir.rstrip("/"))
        agg_csv_path = os.path.join(base_dir, "base_eval", ckpt_subdir,
                                    f"gpt2_aggregate_{ckpt_subdir}.csv")
        os.makedirs(os.path.dirname(agg_csv_path), exist_ok=True)
        # Collect all keys across all steps
        all_keys = []
        for r in aggregate_results:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(agg_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(aggregate_results)
        print0(f"\nAggregate results written to: {agg_csv_path}")

    wb.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
