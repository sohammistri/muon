"""
Download and load FLORES+ en-hi evaluation dataset.

Downloads the devtest split of openlanguagedata/flores_plus, filters for
English-Hindi parallel pairs (aligned by sentence ID), and saves them as
a parquet file under ~/.cache/muon/flores_plus_hi/.

Usage:
    # Download only
    python -m attention_NMT.eval_dataset

    # Use in code
    from attention_NMT.eval_dataset import load_flores_eval
    pairs = load_flores_eval()  # list of {"src": en, "tgt": hi}
"""

import os
import sys
import argparse

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from GPT2.common import get_base_dir


DATA_DIR = os.path.join(get_base_dir(), "flores_plus_hi")
PARQUET_PATH = os.path.join(DATA_DIR, "devtest_en_hi.parquet")


def download_flores_eval():
    """
    Download FLORES+ devtest en-hi pairs and save as a local parquet file.
    Skips download if the file already exists.

    Returns the path to the saved parquet file.
    """
    if os.path.exists(PARQUET_PATH):
        table = pq.read_table(PARQUET_PATH)
        print(f"Already downloaded: {PARQUET_PATH} ({len(table)} pairs)")
        return PARQUET_PATH

    print("Downloading openlanguagedata/flores_plus devtest split...")
    from datasets import load_dataset

    ds = load_dataset("openlanguagedata/flores_plus", split="devtest")

    # Filter English and Hindi rows
    eng_rows = {r["id"]: r["text"] for r in ds if r["iso_639_3"] == "eng"}
    hin_rows = {r["id"]: r["text"] for r in ds if r["iso_639_3"] == "hin"}

    # Align by ID
    common_ids = sorted(eng_rows.keys() & hin_rows.keys())
    if not common_ids:
        raise RuntimeError("No overlapping en-hi sentence IDs found in FLORES+")

    src_texts = [eng_rows[i] for i in common_ids]
    tgt_texts = [hin_rows[i] for i in common_ids]

    # Save as parquet
    os.makedirs(DATA_DIR, exist_ok=True)
    table = pa.table({"id": common_ids, "src": src_texts, "tgt": tgt_texts})
    pq.write_table(table, PARQUET_PATH)

    print(f"Saved {len(common_ids)} en-hi pairs to {PARQUET_PATH}")
    return PARQUET_PATH


def load_flores_eval():
    """
    Load FLORES+ en-hi eval pairs. Downloads if not already cached.

    Returns:
        list[dict] with keys "id", "src" (English), "tgt" (Hindi)
    """
    if not os.path.exists(PARQUET_PATH):
        download_flores_eval()

    table = pq.read_table(PARQUET_PATH)
    return table.to_pylist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FLORES+ en-hi eval dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    if args.force and os.path.exists(PARQUET_PATH):
        os.remove(PARQUET_PATH)
        print("Removed cached file, re-downloading...")

    download_flores_eval()

    # Quick verification
    pairs = load_flores_eval()
    print(f"\nLoaded {len(pairs)} pairs. Sample:")
    print(f"  EN: {pairs[0]['src'][:100]}...")
    print(f"  HI: {pairs[0]['tgt'][:100]}...")
