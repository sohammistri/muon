"""
Upload completed GPT-2 training logs (metrics.jsonl) to W&B project "muon".

Run mapping (verified against config.json files):
  d12-muon-llm-lr3e-4  ← muon-llm-bf16_20260327_081542
  d12-muon-llm-lr1e-3  ← muon-llm-bf16_20260327_112457
  d12-adamw-lr1e-3     ← adamw-bf16_20260327_131643
  d12-adamw-lr3e-4     ← adamw-bf16_20260327_161413
  d24-adamw-lr1e-3     ← adamw-bf16_20260327_191652
  d24-adamw-lr3e-3     ← adamw-bf16_20260327_232526

Usage:
  uv run python GPT2/upload_logs_to_wandb.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import wandb

# ---------------------------------------------------------------------------
# Run mapping: folder-name-suffix → hyperparams
# NOTE: d24-muon-llm run (20260328_033305) is still in progress — excluded.
# ---------------------------------------------------------------------------
LOGS_DIR = Path(__file__).parent / "eval_logs"

RUN_MAPPING: dict[str, dict] = {
    # "metrics_gpt2-eval-d24-muon-llm-lr1e-3_20260328_173552": {"depth": 24, "optim": "muon-llm", "lr": 1e-3},
    "metrics_gpt2-eval-d24-muon-llm-lr3e-3_20260328_190234": {"depth": 24, "optim": "muon-llm", "lr": 3e-3},
    # "metrics_gpt2-eval-d24-adamw-lr1e-3_20260328_144203":    {"depth": 24, "optim": "adamw",    "lr": 1e-3},
    # "metrics_gpt2-eval-d24-adamw-lr3e-3_20260328_160908":    {"depth": 24, "optim": "adamw",    "lr": 3e-3},
    # "metrics_gpt2-adamw-bf16_20260327_191652":    {"depth": 24, "optim": "adamw",    "lr": 1e-3},
    # "metrics_gpt2-adamw-bf16_20260327_232526":    {"depth": 24, "optim": "adamw",    "lr": 3e-3},
    # "metrics_gpt2-muon-llm-bf16_20260328_134543":    {"depth": 24, "optim": "muon-llm",    "lr": 1e-2},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_lr(lr: float) -> str:
    """Convert float lr to compact scientific notation string.

    Examples: 0.001 → '1e-3', 0.0003 → '3e-4', 0.003 → '3e-3'.
    """
    s = f"{lr:.1e}"              # '3.0e-04'
    mantissa, exp_str = s.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")   # '3'
    exp = int(exp_str)                             # -4
    return f"{mantissa}e{exp}"


def make_run_name(depth: int, optim: str, lr: float) -> str:
    return f"d{depth}-{optim}-lr{format_lr(lr)}"


def convert_value(v):
    """Recursively convert W&B histogram dicts to wandb.Histogram objects."""
    if isinstance(v, dict):
        if v.get("_type") == "histogram":
            counts = np.array(v["counts"], dtype=np.int64)
            bin_edges = np.array(v["bin_edges"], dtype=np.float64)
            return wandb.Histogram(np_histogram=(counts, bin_edges))
        return {k: convert_value(val) for k, val in v.items()}
    return v


# ---------------------------------------------------------------------------
# Core upload
# ---------------------------------------------------------------------------

def upload_run(folder_name: str, meta: dict) -> None:
    folder = LOGS_DIR / folder_name
    config_path = folder / "config.json"
    metrics_path = folder / "metrics.jsonl"

    depth  = meta["depth"]
    optim  = meta["optim"]
    lr     = meta["lr"]
    lr_str = format_lr(lr)

    run_name = make_run_name(depth, optim, lr)
    group    = f"d{depth}"
    tags     = [f"d{depth}", optim, f"lr={lr_str}", "bf16"]

    with open(config_path) as f:
        config = json.load(f)

    print(f"\n→ Uploading {run_name}  ({folder_name})")

    run = wandb.init(
        project="muon-eval",
        name=run_name,
        config=config,
        group=group,
        tags=tags,
        resume="never",
    )

    try:
        total_lines = 0
        with open(metrics_path) as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                row = json.loads(raw_line)
                step = row.pop("_step")
                row = convert_value(row)
                wandb.log(row, step=step)
                total_lines += 1

        print(f"  ✓ Logged {total_lines} rows  →  {run.url}")
    finally:
        wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for folder_name, meta in RUN_MAPPING.items():
        folder = LOGS_DIR / folder_name
        if not folder.exists():
            print(f"  ⚠  Folder not found, skipping: {folder}")
            continue
        upload_run(folder_name, meta)

    print("\nDone — all runs uploaded to W&B project 'muon'.")


if __name__ == "__main__":
    main()
