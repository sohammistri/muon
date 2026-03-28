"""
Evaluate compression ratio of the NMT tokenizer on English and Hindi text.
"""

import os
import sys
import argparse

# Add parent directory so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attention_NMT.tokenizer import get_tokenizer, RustBPETokenizer
from attention_NMT.dataset import parquets_iter_batched
from common.logger import setup_logger

parser = argparse.ArgumentParser(description='Evaluate NMT tokenizer compression')
parser.add_argument('--segment', type=str, default='hi', help='Language segment code (default: hi)')
args = parser.parse_args()

logger = setup_logger("tok_eval", "tokenizer", "nmt_eval")

# English news text
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico's National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation's food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

"The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening's to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border," said U.S. Secretary of Agriculture Brooke L. Rollins. "Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest."
""".strip()

# Hindi news text
hindi_news_text = r"""
नई दिल्ली, 15 जुलाई 2025 — भारत सरकार ने आज एक नई शिक्षा नीति की घोषणा की जिसका उद्देश्य देश भर में डिजिटल शिक्षा को बढ़ावा देना है। शिक्षा मंत्रालय के अनुसार, इस नीति के तहत सभी सरकारी स्कूलों में उच्च गति इंटरनेट कनेक्टिविटी प्रदान की जाएगी और शिक्षकों को डिजिटल उपकरणों के उपयोग में प्रशिक्षित किया जाएगा।

मंत्रालय के प्रवक्ता ने कहा, "हमारा लक्ष्य 2030 तक हर बच्चे को गुणवत्तापूर्ण डिजिटल शिक्षा उपलब्ध कराना है। यह नीति ग्रामीण और शहरी क्षेत्रों के बीच शिक्षा की खाई को पाटने में महत्वपूर्ण भूमिका निभाएगी।"

विशेषज्ञों ने इस पहल की सराहना करते हुए कहा कि यह भारत के विकास में एक महत्वपूर्ण कदम है, हालांकि कार्यान्वयन की चुनौतियों पर भी ध्यान देने की आवश्यकता है।
""".strip()

# Code text
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

# Math text (LaTeX)
math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}

\begin{document}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.
Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then $S(n+1) = S(n) + (n+1)^3 = \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3$.
Factor out $(n+1)^2$ to get $S(n+1) = \left(\frac{(n+1)(n+2)}{2}\right)^2$.
\end{proof}

\end{document}
""".strip()

# Science text
science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates.
""".strip()

# Load train/val data from samanantar (both src and tgt)
train_batch = next(parquets_iter_batched(args.segment, split="train"))
train_src, train_tgt = train_batch
train_text = "\n".join(train_src + train_tgt)

val_batch = next(parquets_iter_batched(args.segment, split="val"))
val_src, val_tgt = val_batch
val_text = "\n".join(val_src + val_tgt)

all_text = [
    ("en_news", news_text),
    ("hi_news", hindi_news_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
    ("sam-train", train_text),
]
if val_text:
    all_text.append(("sam-val", val_text))

# Compare our NMT tokenizer against GPT-2 and GPT-4 tokenizers
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:

    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2")
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base")
    else:
        tokenizer = get_tokenizer(args.segment)

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': ratio
        }

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Log vocab sizes
logger.info(f"Vocab sizes: GPT-2={vocab_sizes['gpt2']}, GPT-4={vocab_sizes['gpt4']}, Ours={vocab_sizes['ours']}")

def log_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Log comparison table between baseline tokenizer and ours."""
    logger.info(f"Comparison with {baseline_name}:")
    header = f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}"
    subhdr = f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}"
    logger.info("=" * 95)
    logger.info(header)
    logger.info(subhdr)
    logger.info("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        logger.info(f"{name:<10} {baseline_data['bytes']:<8} "
                    f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
                    f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
                    f"{ours_color}{ours_data['tokens']:<7}{RESET} "
                    f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
                    f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
                    f"{better:<10}")

# Log comparisons
log_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
log_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)
logger.info("Done!")
