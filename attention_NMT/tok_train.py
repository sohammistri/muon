"""
Train a shared BPE tokenizer for NMT on both source (English) and target language sentences.
In the style of GPT-4 tokenizer.
"""
import os
import sys
import time
import argparse
import torch

# Add parent directory so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attention_NMT.tokenizer import RustBPETokenizer
from attention_NMT.dataset import parquets_iter_batched
from GPT2.common import get_base_dir
from common.logger import setup_logger

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a shared BPE tokenizer for NMT')
parser.add_argument('--segment', type=str, default='hi', help='Language segment code (default: hi)')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per sentence (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()

logger = setup_logger("tok_train", "tokenizer", "nmt_bpe")
logger.info(f"segment: {args.segment}")
logger.info(f"max_chars: {args.max_chars:,}")
logger.info(f"doc_cap: {args.doc_cap:,}")
logger.info(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator — yields individual sentences from both src and tgt

def text_iterator():
    """
    Iterate over samanantar parallel corpus, yielding individual sentences
    from both src (English) and tgt (target language) for shared BPE training.
    """
    nchars = 0
    for src_batch, tgt_batch in parquets_iter_batched(args.segment, split="train"):
        for sentence in src_batch + tgt_batch:
            text = sentence
            if len(text) > args.doc_cap:
                text = text[:args.doc_cap]
            nchars += len(text)
            yield text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
logger.info(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, f"nmt_tokenizer_{args.segment}")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check — test on both English and target language
test_en = "Hello world! This is a test sentence for the NMT tokenizer."
encoded_en = tokenizer.encode(test_en)
decoded_en = tokenizer.decode(encoded_en)
assert decoded_en == test_en, f"English roundtrip failed: {decoded_en!r} != {test_en!r}"

test_hi = "नमस्ते दुनिया! यह एनएमटी टोकनाइज़र के लिए एक परीक्षण वाक्य है।"
encoded_hi = tokenizer.encode(test_hi)
decoded_hi = tokenizer.decode(encoded_hi)
assert decoded_hi == test_hi, f"Hindi roundtrip failed: {decoded_hi!r} != {test_hi!r}"

logger.info(f"Sanity check passed: EN '{test_en[:40]}...' -> {len(encoded_en)} tokens")
logger.info(f"Sanity check passed: HI '{test_hi[:40]}...' -> {len(encoded_hi)} tokens")

# -----------------------------------------------------------------------------
# Cache token-id -> byte-count mapping for bits-per-byte evaluation
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]
    if token_str in special_set:
        token_bytes.append(0)
    else:
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
logger.info(f"Saved token_bytes to {token_bytes_path}")

# Log token byte stats
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
logger.info(f"Token bytes stats (non-special): min={int(token_bytes_nonzero.min().item())}, "
            f"max={int(token_bytes_nonzero.max().item())}, "
            f"mean={token_bytes_nonzero.mean().item():.2f}, "
            f"std={token_bytes_nonzero.std().item():.2f}")
logger.info(f"num_special_tokens: {len(special_set)}")
