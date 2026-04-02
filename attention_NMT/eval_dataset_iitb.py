"""
Download and load IITB English-Hindi test dataset.

Downloads the test split of cfilt/iitb-english-hindi, extracts parallel
English-Hindi pairs, and saves them as a parquet file under
~/.cache/muon/iitb_en_hi/.

Usage:
    # Download only
    python -m attention_NMT.eval_dataset_iitb

    # Use in code
    from attention_NMT.eval_dataset_iitb import load_iitb_eval
    pairs = load_iitb_eval()  # list of {"src": en, "tgt": hi}
"""

import os
import sys
import argparse

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from GPT2.common import get_base_dir


DATA_DIR = os.path.join(get_base_dir(), "iitb_en_hi")
PARQUET_PATH = os.path.join(DATA_DIR, "test_en_hi.parquet")


def download_iitb_eval():
    """
    Download IITB en-hi test pairs and save as a local parquet file.
    Skips download if the file already exists.

    Returns the path to the saved parquet file.
    """
    if os.path.exists(PARQUET_PATH):
        table = pq.read_table(PARQUET_PATH)
        print(f"Already downloaded: {PARQUET_PATH} ({len(table)} pairs)")
        return PARQUET_PATH

    print("Downloading cfilt/iitb-english-hindi test split...")
    from datasets import load_dataset

    ds = load_dataset("cfilt/iitb-english-hindi", split="test")

    src_texts = [row["translation"]["en"] for row in ds]
    tgt_texts = [row["translation"]["hi"] for row in ds]

    # Save as parquet
    os.makedirs(DATA_DIR, exist_ok=True)
    table = pa.table({"src": src_texts, "tgt": tgt_texts})
    pq.write_table(table, PARQUET_PATH)

    print(f"Saved {len(src_texts)} en-hi pairs to {PARQUET_PATH}")
    return PARQUET_PATH


def load_iitb_eval():
    """
    Load IITB en-hi eval pairs. Downloads if not already cached.

    Returns:
        list[dict] with keys "src" (English), "tgt" (Hindi)
    """
    if not os.path.exists(PARQUET_PATH):
        download_iitb_eval()

    table = pq.read_table(PARQUET_PATH)
    return table.to_pylist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IITB en-hi test dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    if args.force and os.path.exists(PARQUET_PATH):
        os.remove(PARQUET_PATH)
        print("Removed cached file, re-downloading...")

    download_iitb_eval()

    # Quick verification
    pairs = load_iitb_eval()
    print(f"\nLoaded {len(pairs)} pairs. Sample:")
    print(f"  EN: {pairs[0]['src'][:100]}...")
    print(f"  HI: {pairs[0]['tgt'][:100]}...")
