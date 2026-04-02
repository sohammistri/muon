"""
Interactive CLI for English → Hindi NMT translation.

Usage:
    python -m attention_NMT.translate --checkpoint-dir attention_NMT/checkpoints
    python -m attention_NMT.translate --checkpoint-dir attention_NMT/checkpoints --step 5000
    python -m attention_NMT.translate --checkpoint-dir attention_NMT/checkpoints --device cpu --precision bf16
"""

import argparse
import contextlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attention_NMT.eval import load_nmt_model, greedy_decode, ids_to_text
from attention_NMT.tokenizer import get_tokenizer
from GPT2.checkpoint import find_latest_step
from GPT2.common import autodetect_device_type


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive English → Hindi NMT translator")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory containing model_*.pt and meta_*.json checkpoint files")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step to load (default: latest)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on: cuda, mps, or cpu (default: autodetect)")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                        help="Inference precision (default: fp32)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Resolve device ---
    device_type = args.device if args.device else autodetect_device_type()
    device = torch.device(device_type)

    # --- Resolve checkpoint step ---
    checkpoint_dir = args.checkpoint_dir
    if args.step is not None:
        step = args.step
    else:
        step = find_latest_step(checkpoint_dir)
        if step is None:
            print(f"Error: no checkpoints found in '{checkpoint_dir}'", file=sys.stderr)
            sys.exit(1)

    # --- Load tokenizer ---
    tokenizer = get_tokenizer("hi")
    bos_id = tokenizer.get_bos_token_id()
    eos_id = tokenizer.get_eos_token_id()
    pad_id = tokenizer.get_vocab_size()

    # --- Load model ---
    print(f"Loading checkpoint: {checkpoint_dir}  (step {step})")
    model, context_window, loaded_step = load_nmt_model(checkpoint_dir, step, device)
    print(f"Model loaded  |  step={loaded_step}  device={device_type}  precision={args.precision}")

    # --- Precision context ---
    if args.precision == "bf16":
        precision_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        precision_ctx = torch.amp.autocast(device_type=device_type, enabled=False)

    print("\nType an English sentence and press Enter to translate. Ctrl-D or empty line to quit.\n")

    # --- Interactive REPL ---
    while True:
        try:
            sentence = input("en> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if not sentence:
            print("Goodbye.")
            sys.exit(0)

        src_tokens = tokenizer.encode(sentence, prepend=bos_id, append=eos_id)
        src_ids = torch.tensor([src_tokens], dtype=torch.long, device=device)
        src_pad_mask = (src_ids != pad_id)

        decoded = greedy_decode(
            model, src_ids, src_pad_mask,
            bos_id, eos_id, pad_id,
            max_len=context_window,
            precision_ctx=precision_ctx,
        )
        translation = ids_to_text(decoded[0], tokenizer, bos_id, eos_id, pad_id)
        print(f"hi> {translation}\n")


if __name__ == "__main__":
    main()
