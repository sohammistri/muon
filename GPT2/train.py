import argparse
import contextlib
import math
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from muon import MuonJordan, MuonLLM
from GPT2.model import GPT2
from GPT2.data import get_pretraining_loaders
from GPT2.common import (
    compute_init,
    compute_cleanup,
    get_dist_info,
    autodetect_device_type,
    print0,
    print_banner,
    get_peak_flops,
    DummyWandb,
    LocalMetricLogger,
)
from common.logger import setup_logger
from common.metrics import compute_weight_diagnostics, compute_gradient_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 Pretraining Benchmark")

    # Model architecture
    parser.add_argument("--emb_dim", type=int, default=768,
                        help="Embedding dimension")
    parser.add_argument("--context_window", type=int, default=1024,
                        help="Sequence length (T)")
    parser.add_argument("--depth", type=int, default=12,
                        help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=None,
                        help="Number of attention heads (default: emb_dim // 64)")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--optim", choices=["adamw", "muon-jordan", "muon-llm"],
                        default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Sequences per batch")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm (0 to disable)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Linear LR warmup steps")

    # Precision
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp8"],
                        default="bf16",
                        help="Training precision (fp8 requires torchao)")

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N training steps")
    parser.add_argument("--val_steps", type=int, default=20,
                        help="Number of validation batches per eval")

    # Misc
    parser.add_argument("--log_diagnostics", action="store_true",
                        help="Log detailed optimizer diagnostics to W&B")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable torch.compile (--no-compile to disable)")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable W&B logging (--no-wandb to disable)")

    args = parser.parse_args()

    # --- Validations ---
    if args.num_heads is None:
        if args.emb_dim % 64 != 0:
            parser.error(
                f"emb_dim ({args.emb_dim}) must be divisible by 64 for default "
                f"head_dim=64, or provide --num_heads explicitly"
            )
        args.num_heads = args.emb_dim // 64

    if args.emb_dim % args.num_heads != 0:
        parser.error(
            f"emb_dim ({args.emb_dim}) must be divisible by "
            f"num_heads ({args.num_heads})"
        )

    if args.context_window <= 0:
        parser.error("context_window must be positive")
    if args.depth <= 0:
        parser.error("depth must be positive")
    if args.max_steps <= 0:
        parser.error("max_steps must be positive")
    if args.warmup_steps >= args.max_steps:
        parser.error(
            f"warmup_steps ({args.warmup_steps}) must be less than "
            f"max_steps ({args.max_steps})"
        )

    if args.precision == "bf16" and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap < (8, 0):
            parser.error(
                f"bf16 requires SM 80+ (Ampere), got SM {cap[0]}{cap[1]}. "
                f"Use --precision fp32"
            )

    if args.precision == "fp8":
        if not torch.cuda.is_available():
            parser.error("fp8 precision requires CUDA")
        try:
            import torchao.float8  # noqa: F401
        except ImportError:
            parser.error(
                "fp8 precision requires torchao. Install with: "
                "pip install torchao"
            )

    return args


class OptimizerGroup:
    """Wraps multiple optimizers with a unified interface."""
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


def create_optimizer(model, args):
    if args.optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Muon: backbone Linear weights get Newton-Schulz; everything else gets AdamW
    modules = dict(model.named_modules())
    muon_params = []
    for name, p in model.named_parameters():
        if "backbone" in name and name.endswith(".weight"):
            mod_name = name.rsplit(".", 1)[0]
            if isinstance(modules[mod_name], nn.Linear):
                muon_params.append(p)

    muon_ids = {id(p) for p in muon_params}
    adam_params = [p for p in model.parameters() if id(p) not in muon_ids]

    MuonCls = MuonJordan if args.optim == "muon-jordan" else MuonLLM

    return OptimizerGroup(
        MuonCls(muon_params, lr=args.lr, momentum=0.95, steps=5,
                weight_decay=args.weight_decay, nesterov=True),
        torch.optim.AdamW(adam_params, lr=args.lr,
                          weight_decay=args.weight_decay),
    )


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=0.0):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay_ratio))


def set_lr(optimizer, lr):
    """Set learning rate on optimizer or OptimizerGroup."""
    if isinstance(optimizer, OptimizerGroup):
        for opt in optimizer.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = lr


@torch.no_grad()
def evaluate(model, val_loader, val_steps, precision_ctx):
    model.eval()
    total_loss = 0.0
    for _ in range(val_steps):
        inputs, targets = next(val_loader)
        with precision_ctx:
            logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    model.train()
    avg_loss = total_loss / val_steps
    return {
        "eval/loss": avg_loss,
        "eval/perplexity": math.exp(min(avg_loss, 20.0)),  # clamp to avoid overflow
    }


def setup_precision(args, model, device_type):
    """Configure mixed precision and return (model, autocast_ctx).

    For fp8, converts model Linear layers AFTER optimizer creation.
    """
    if args.precision == "bf16":
        ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    elif args.precision == "fp8":
        from torchao.float8 import convert_to_float8_training
        convert_to_float8_training(model)
        # fp8 linear layers handle fp8 compute; bf16 autocast for non-linear ops
        ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        ctx = contextlib.nullcontext()
    return ctx


def main():
    args = parse_args()
    print_banner()

    # Device and DDP setup
    device_type = args.device if args.device else autodetect_device_type()
    use_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    log = setup_logger("gpt2", "pretrain", args.optim)
    if ddp_rank == 0:
        log.info(f"Device: {device} | DDP: {use_ddp} (world_size={ddp_world_size})")
        log.info(f"Precision: {args.precision}")

    # Data
    train_loader, val_loader, vocab_size = get_pretraining_loaders(
        B=args.batch_size, T=args.context_window, device=device,
    )
    if ddp_rank == 0:
        log.info(f"Vocab size: {vocab_size:,}")

    # Model
    model = GPT2(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        context_window=args.context_window,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    if ddp_rank == 0:
        log.info(
            f"GPT2: emb_dim={args.emb_dim}, heads={args.num_heads}, "
            f"depth={args.depth}, ctx={args.context_window} | "
            f"Parameters: {param_count:,}"
        )

    # Optimizer (BEFORE fp8 conversion so nn.Linear isinstance checks work)
    optimizer = create_optimizer(model, args)

    # Mixed precision setup (fp8 converts model in-place)
    precision_ctx = setup_precision(args, model, device_type)

    # DDP
    if use_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if use_ddp else model

    # Compile
    if args.compile:
        model = torch.compile(model)

    # W&B or local metric logging
    if args.wandb and ddp_rank == 0:
        wandb.init(
            project="muon",
            name=f"gpt2-{args.optim}-{args.precision}",
            config=vars(args),
        )
        wb = wandb
    elif ddp_rank == 0:
        wb = LocalMetricLogger(
            project="muon",
            name=f"gpt2-{args.optim}-{args.precision}",
            config=vars(args),
        )
    else:
        wb = DummyWandb()

    # MFU setup
    if device_type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        peak_flops = get_peak_flops(device_name)
    else:
        peak_flops = float("inf")
    tokens_per_step = args.batch_size * args.context_window * ddp_world_size
    tokens_per_gpu_step = args.batch_size * args.context_window

    # Step-0 evaluation
    global_step = 0
    metrics = evaluate(model, val_loader, args.val_steps, precision_ctx)
    if args.log_diagnostics:
        weight_diag = compute_weight_diagnostics(raw_model)
        metrics.update(weight_diag)
        for k, v in list(metrics.items()):
            if k.endswith("/sv_histogram"):
                metrics[k] = wandb.Histogram(v)
    wb.log(metrics, step=global_step)
    if ddp_rank == 0:
        summary = ", ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items()
            if not k.startswith("diagnostics/")
        )
        log.info(f"[Step {global_step}] {summary}")

    # Training loop
    model.train()
    t0 = time.perf_counter()

    for step in range(1, args.max_steps + 1):
        # LR schedule
        lr = get_lr(step - 1, args.warmup_steps, args.max_steps, args.lr)
        set_lr(optimizer, lr)

        # Forward
        inputs, targets, data_state = next(train_loader)
        optimizer.zero_grad()

        with precision_ctx:
            logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Backward
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Timing
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = t1 - t0
        t0 = t1

        tokens_per_sec = tokens_per_step / dt
        flops_per_step = 6 * param_count * tokens_per_gpu_step
        mfu = flops_per_step / (peak_flops * dt)

        # Log training metrics
        train_metrics = {
            "train/loss": loss.item(),
            "train/lr": lr,
            "train/tokens_per_sec": tokens_per_sec,
            "train/mfu": mfu,
        }
        wb.log(train_metrics, step=step)

        if ddp_rank == 0 and step % 10 == 0:
            log.info(
                f"[Step {step}/{args.max_steps}] loss={loss.item():.4f} "
                f"lr={lr:.2e} tok/s={tokens_per_sec:.0f} mfu={mfu:.2%}"
            )

        # Evaluation
        if step % args.eval_every == 0:
            if args.log_diagnostics:
                grad_diag = compute_gradient_diagnostics(raw_model)
                weight_diag = compute_weight_diagnostics(raw_model)
            metrics = evaluate(model, val_loader, args.val_steps, precision_ctx)
            if args.log_diagnostics:
                metrics.update(grad_diag)
                metrics.update(weight_diag)
                for k, v in list(metrics.items()):
                    if k.endswith("/sv_histogram"):
                        metrics[k] = wandb.Histogram(v)
            wb.log(metrics, step=step)
            if ddp_rank == 0:
                summary = ", ".join(
                    f"{k}: {v:.4f}" for k, v in metrics.items()
                    if not k.startswith("diagnostics/")
                )
                log.info(f"[Step {step}] {summary}")

    # Final evaluation
    if args.log_diagnostics:
        grad_diag = compute_gradient_diagnostics(raw_model)
        weight_diag = compute_weight_diagnostics(raw_model)
    metrics = evaluate(model, val_loader, args.val_steps, precision_ctx)
    if args.log_diagnostics:
        metrics.update(grad_diag)
        metrics.update(weight_diag)
        for k, v in list(metrics.items()):
            if k.endswith("/sv_histogram"):
                metrics[k] = wandb.Histogram(v)
    wb.log(metrics, step=args.max_steps)
    if ddp_rank == 0:
        summary = ", ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items()
            if not k.startswith("diagnostics/")
        )
        log.info(f"Final eval: {summary}")

    wb.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
