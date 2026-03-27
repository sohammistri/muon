# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo implements the **Muon optimizer** (Newton-Schulz orthogonalization-based optimizer) and benchmarks it against SGD and AdamW on MLP, CNN, and GPT-2 pretraining tasks. Experiment tracking uses Weights & Biases.

## Commands

```bash
# Install dependencies (uses uv with Python 3.13)
uv sync

# Run MLP benchmark (main entry point)
uv run python MLP/train.py --optim muon --dataset mnist --epochs 10

# Run with diagnostics (SVD metrics, gradient norms logged to W&B)
uv run python MLP/train.py --optim muon --dataset covertype --log_diagnostics

# Available datasets: covertype (classification), year_prediction (regression), mnist (classification)
# Available optimizers: sgd, adamw, muon-jordan, muon-llm

# Run CNN benchmark
uv run python CNN/train.py --optim muon-jordan --dataset cifar10 --epochs 20

# Run CNN with diagnostics
uv run python CNN/train.py --optim muon-llm --dataset cifar10 --log_diagnostics

# Available CNN datasets: cifar10 (classification)

# Run GPT-2 pretraining benchmark
uv run python GPT2/train.py --optim muon-llm --precision bf16 --max_steps 10000

# Run GPT-2 with smaller model (quick test)
uv run python GPT2/train.py --optim adamw --precision fp32 --device cpu --max_steps 5 \
    --batch_size 2 --context_window 64 --emb_dim 64 --depth 2 --warmup_steps 2 --no-wandb

# Run GPT-2 with diagnostics
uv run python GPT2/train.py --optim muon-jordan --log_diagnostics

# Available GPT-2 optimizers: adamw, muon-jordan, muon-llm (no sgd)
# Available precisions: fp32, bf16, fp8 (fp8 requires torchao + Hopper GPU)
```

## Architecture

- **`muon.py`** — Core Muon optimizer. Uses Newton-Schulz iteration (`zeropower_via_newtonschulz5`) to approximate orthogonal projection of gradient matrices. Falls back to Adam for non-matrix params (biases, BatchNorm). The NS iteration uses quintic coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)` optimized for maximal slope at zero.

- **`MLP/train.py`** — Training script and CLI entry point. When using Muon, wraps it with `OptimizerGroup` that pairs Muon (for backbone Linear weights) with AdamW (for head, biases, BatchNorm params). This split is important — only 2D backbone Linear weights go through Newton-Schulz.

- **`MLP/model.py`** — Generic MLP with configurable hidden dims. Uses BatchNorm + GELU + Dropout blocks in the backbone, separate Linear head.

- **`MLP/data.py`** — Data loaders for three datasets: Forest Covertype (sklearn), Year Prediction MSD (UCI download), MNIST (torchvision). All apply StandardScaler normalization. Cached in `MLP/data_cache/`.

- **`CNN/train.py`** — CNN training script and CLI entry point. Same OptimizerGroup pattern as MLP — backbone Conv2d and Linear weights go through Newton-Schulz, head/biases/BatchNorm go to AdamW. Muon handles 4D conv weights by reshaping to 2D internally.

- **`CNN/model.py`** — Simple CNN with configurable channel widths. Backbone uses Conv2d-BN-GELU blocks with MaxPool2d, separate Linear head. AdaptiveAvgPool2d bridges backbone to head.

- **`CNN/data.py`** — CIFAR-10 data loader with standard augmentation (RandomCrop, RandomHorizontalFlip). Cached in `CNN/data_cache/`.

- **`GPT2/train.py`** — GPT-2 pretraining script. Step-based training loop (not epoch-based) with cosine LR decay + linear warmup. Supports mixed precision (`fp32`, `bf16`, `fp8`). Same OptimizerGroup/Muon param splitting as MLP/CNN — backbone Linear weights (qkv_proj, out_proj, FFN) go through Newton-Schulz, everything else (Embedding, LayerNorm, head) goes to AdamW. Logs MFU and tokens/sec. Optimizer is created BEFORE fp8 conversion to preserve `nn.Linear` isinstance checks.

- **`GPT2/model.py`** — GPT-2 decoder-only transformer. `CausalSelfAttention` with fused QKV projection and `F.scaled_dot_product_attention`. `TransformerBlock` with pre-norm (LayerNorm → Attention/FFN). Sinusoidal positional embeddings (buffer, not learned). Backbone (`ModuleDict`) / head (`nn.Linear`) split matches MLP/CNN convention.

- **`GPT2/data.py`** — Wraps `GPT2/dataloader.py` with a clean API: `get_pretraining_loaders(B, T, device)` returns infinite generators `(train_loader, val_loader, vocab_size)`. Train loader yields `(inputs, targets, state_dict)`, val loader yields `(inputs, targets)`. Data is pre-moved to device.

- **`GPT2/dataloader.py`** — BOS-aligned best-fit document packing. Every row starts with BOS token, documents packed via best-fit algorithm to minimize cropping (~35% tokens cropped at T=2048). 100% utilization (no padding). Handles DDP sharding internally. Uses pinned memory + single HtoD transfer.

- **`GPT2/tokenizer.py`** — BPE tokenizer with two backends: `HuggingFaceTokenizer` and `RustBPETokenizer` (rustbpe for training + tiktoken for inference). GPT-4-style split pattern. Special tokens include `<|bos|>` and conversation delimiters.

- **`GPT2/dataset.py`** — Dataset download and parquet file management for pretraining data.

- **`GPT2/common.py`** — Shared utilities: DDP setup (`compute_init`/`compute_cleanup`/`get_dist_info`), device auto-detection, peak FLOPS table for MFU calculation, `DummyWandb` for non-rank-0 processes.

- **`common/metrics.py`** — SVD-based weight diagnostics (condition number, effective rank, spectral norm, orthogonality error) and gradient diagnostics. These are the key metrics for understanding Muon's effect on weight matrices.

- **`common/logger.py`** — Colored console + file logging. Logs saved to `logs/`.

## Testing

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run MLP tests only (38 tests)
uv run python -m pytest tests/test_mlp.py -v

# Run CNN tests only (36 tests)
uv run python -m pytest tests/test_cnn.py -v

# Run GPT2 tests only (54 tests)
uv run python -m pytest tests/test_gpt2.py -v
```

- **`tests/test_mlp.py`** — 38 tests covering MLP model construction, forward pass (classification + regression squeeze), data loaders (covertype, mnist, year_prediction), optimizer creation and Muon param splitting (2D backbone Linear weights only), training convergence for all 4 optimizers on both classification and regression, evaluate metrics, and SVD-based diagnostics.

- **`tests/test_cnn.py`** — 36 tests covering CNN model construction (including AdaptiveAvgPool2d variable spatial input), CIFAR-10 data loading, optimizer creation and Muon param splitting (Conv2d 4D + Linear 2D weights), 4D conv weight shape preservation after Muon steps, training convergence, evaluate metrics (classification-only, no regression keys), weight decay verification, and diagnostics (condition number, spectral norm, orthogonality error, effective rank).

- **`tests/test_gpt2.py`** — 54 tests covering GPT2 model construction and forward pass, CausalSelfAttention (output shape, causal masking verification), TransformerBlock residual connections, sinusoidal PE (shape, buffer-not-parameter), optimizer creation and Muon param splitting (2D backbone Linear weights only, excludes Embedding/LayerNorm/head), LR schedule (linear warmup, cosine decay, set_lr on OptimizerGroup), mixed precision setup, training convergence for all 3 optimizers, gradient flow, weight shape preservation after Muon steps, gradient clipping, evaluate (loss + perplexity, no accuracy/regression keys, no gradients, restores train mode), and SVD-based diagnostics.

### Test patterns
- Tests use small models (`channels=[8,16]`, `hidden_dims=[32,16]`, `emb_dim=64, depth=2`) and tiny batches (B=2–8) for speed.
- `_Args` helper class mocks the argparse namespace for `create_optimizer`.
- Evaluate tests use `TensorDataset` fake loaders (MLP/CNN) or infinite generators (GPT2) to avoid dataset downloads.
- Data loader tests download datasets on first run (CIFAR-10, Covertype, MNIST, Year Prediction MSD).
- GPT2 evaluate uses an infinite generator yielding `(inputs, targets)` matching the real data pipeline API, not a finite DataLoader.
