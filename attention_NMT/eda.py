"""EDA on ai4bharat/samanantar en-hi parallel corpus."""
import pandas as pd
import numpy as np
from pathlib import Path
import re

DATA_DIR = Path.home() / ".cache" / "muon" / "samanantar"

# Load all 5 parquet shards
files = sorted(DATA_DIR.glob("*.parquet"))
print(f"Found {len(files)} parquet files:")
for f in files:
    print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"\n{'='*60}")
print(f"TOTAL ROWS: {len(df):,}")
print(f"COLUMNS: {list(df.columns)}")
print(f"DTYPES:\n{df.dtypes}")
print(f"\nNULL COUNTS:\n{df.isnull().sum()}")
print(f"\nDUPLICATES (full row): {df.duplicated().sum():,}")
print(f"DUPLICATE src: {df['src'].duplicated().sum():,}")
print(f"DUPLICATE tgt: {df['tgt'].duplicated().sum():,}")

# --- Length analysis ---
df["src_len_chars"] = df["src"].str.len()
df["tgt_len_chars"] = df["tgt"].str.len()
df["src_len_words"] = df["src"].str.split().str.len()
df["tgt_len_words"] = df["tgt"].str.split().str.len()
df["len_ratio"] = df["tgt_len_chars"] / df["src_len_chars"].clip(lower=1)

print(f"\n{'='*60}")
print("SOURCE (English) LENGTH STATS (chars):")
print(df["src_len_chars"].describe().to_string())
print(f"\nTARGET (Hindi) LENGTH STATS (chars):")
print(df["tgt_len_chars"].describe().to_string())

print(f"\nSOURCE (English) LENGTH STATS (words):")
print(df["src_len_words"].describe().to_string())
print(f"\nTARGET (Hindi) LENGTH STATS (words):")
print(df["tgt_len_words"].describe().to_string())

print(f"\nTGT/SRC LENGTH RATIO (chars):")
print(df["len_ratio"].describe().to_string())

# --- Percentiles ---
for col, label in [("src_len_words", "Source words"), ("tgt_len_words", "Target words")]:
    pcts = df[col].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).astype(int)
    print(f"\n{label} percentiles:")
    for p, v in pcts.items():
        print(f"  p{int(p*100):02d}: {v}")

# --- Empty / very short sentences ---
empty_src = (df["src_len_words"] == 0).sum()
empty_tgt = (df["tgt_len_words"] == 0).sum()
short_src = (df["src_len_words"] <= 2).sum()
short_tgt = (df["tgt_len_words"] <= 2).sum()
print(f"\nEmpty src: {empty_src:,}  |  Empty tgt: {empty_tgt:,}")
print(f"Very short (<=2 words) src: {short_src:,}  |  tgt: {short_tgt:,}")

# --- Very long sentences ---
long_src = (df["src_len_words"] > 200).sum()
long_tgt = (df["tgt_len_words"] > 200).sum()
print(f"Very long (>200 words) src: {long_src:,}  |  tgt: {long_tgt:,}")

# --- Sample sentences ---
print(f"\n{'='*60}")
print("SAMPLE PAIRS (5 random):")
for _, row in df.sample(5, random_state=42).iterrows():
    print(f"\n  EN: {row['src'][:120]}...")
    print(f"  HI: {row['tgt'][:120]}...")
    print(f"  (src={row['src_len_words']}w, tgt={row['tgt_len_words']}w, ratio={row['len_ratio']:.2f})")

# --- Language mixing check (heuristic: count Devanagari chars in src) ---
devanagari_re = re.compile(r'[\u0900-\u097F]')
df["src_has_hindi"] = df["src"].apply(lambda x: bool(devanagari_re.search(x)))
hindi_in_src = df["src_has_hindi"].sum()
print(f"\n{'='*60}")
print(f"Rows with Devanagari chars in English src: {hindi_in_src:,} ({hindi_in_src/len(df)*100:.2f}%)")

# --- Summary ---
print(f"\n{'='*60}")
print("SUMMARY")
print(f"  Total parallel pairs: {len(df):,}")
print(f"  Avg src length: {df['src_len_words'].mean():.1f} words")
print(f"  Avg tgt length: {df['tgt_len_words'].mean():.1f} words")
print(f"  Median src length: {df['src_len_words'].median():.0f} words")
print(f"  Median tgt length: {df['tgt_len_words'].median():.0f} words")
print(f"  Avg tgt/src char ratio: {df['len_ratio'].mean():.2f}")
print(f"  Duplicates (src): {df['src'].duplicated().sum():,}")
print(f"  Data quality flags: {hindi_in_src:,} rows with Hindi in src")
