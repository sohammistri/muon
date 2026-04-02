"""
Download and iterate over ai4bharat/samanantar parallel corpus parquet files.

Shard metadata (count, URLs) is fetched dynamically from the HuggingFace
Dataset Viewer API — nothing is hardcoded beyond the API endpoint.

Usage:
    python -m attention_NMT.dataset --segment hi -n 5 -w 4
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from GPT2.common import get_base_dir

# -----------------------------------------------------------------------------
# Constants

PARQUET_API_URL = "https://datasets-server.huggingface.co/parquet?dataset=ai4bharat/samanantar"

# -----------------------------------------------------------------------------
# Helpers

def get_data_dir(segment):
    """Returns the local cache directory for a given language segment."""
    return os.path.join(get_base_dir(), f"samanantar_{segment}")


def fetch_parquet_urls(segment):
    """
    Query the HF Dataset Viewer API and return [(filename, url), ...] for the
    requested language segment, sorted by filename.

    Raises ValueError if the segment is not found in the dataset.
    """
    response = requests.get(PARQUET_API_URL, timeout=30)
    response.raise_for_status()
    data = response.json()

    entries = [
        (entry["filename"], entry["url"])
        for entry in data["parquet_files"]
        if entry["config"] == segment
    ]

    if not entries:
        available = sorted({e["config"] for e in data["parquet_files"]})
        raise ValueError(
            f"Segment '{segment}' not found. Available segments: {available}"
        )

    entries.sort(key=lambda x: x[0])
    return entries


# -----------------------------------------------------------------------------
# Utilities importable by other modules

def list_parquet_files(data_dir=None, segment=None):
    """Lists all downloaded parquet files (excluding .tmp), returns sorted full paths."""
    if data_dir is None:
        if segment is None:
            raise ValueError("Either data_dir or segment must be provided")
        data_dir = get_data_dir(segment)

    if not os.path.exists(data_dir):
        return []

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    return [os.path.join(data_dir, f) for f in parquet_files]


def parquets_iter_batched(segment, split="train", start=0, step=1):
    """
    Iterate through the dataset in batches of row groups.
    Yields (src_texts, tgt_texts) tuples where both are list[str].

    - split="train": all parquet files except the last
    - split="val": only the last parquet file
    - start/step: useful for DDP sharding (start=rank, step=world_size)
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(segment=segment)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            src_texts = rg.column('src').to_pylist()
            tgt_texts = rg.column('tgt').to_pylist()
            yield (src_texts, tgt_texts)


# -----------------------------------------------------------------------------
# Download logic

def download_single_file(args):
    """Downloads a single parquet file. Accepts (filename, url, data_dir) tuple."""
    filename, url, data_dir = args

    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filename} (already exists)")
        return True

    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Samanantar parallel corpus shards")
    parser.add_argument("--segment", type=str, required=True,
                        help="Language segment code (e.g., hi, gu, as, bn, ...)")
    parser.add_argument("-n", "--num-files", type=int, default=None,
                        help="Number of shards to download (default: all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4,
                        help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    # Fetch available shards from HF API
    print(f"Fetching shard metadata for segment '{args.segment}'...")
    all_shards = fetch_parquet_urls(args.segment)
    total_available = len(all_shards)

    if args.num_files is None:
        shards_to_download = all_shards
        print(f"No --num-files specified. Downloading all {total_available} shards for segment '{args.segment}'.")
    else:
        shards_to_download = all_shards[:args.num_files]
        print(f"Downloading {len(shards_to_download)}/{total_available} shards for segment '{args.segment}'.")

    # Prepare output directory
    data_dir = get_data_dir(args.segment)
    os.makedirs(data_dir, exist_ok=True)

    # Download
    download_args = [(filename, url, data_dir) for filename, url in shards_to_download]
    print(f"Using {args.num_workers} workers...")
    print(f"Target directory: {data_dir}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, download_args)

    successful = sum(1 for s in results if s)
    print(f"\nDone! Downloaded: {successful}/{len(shards_to_download)} shards to {data_dir}")
