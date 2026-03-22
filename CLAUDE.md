# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo implements the **Muon optimizer** (Newton-Schulz orthogonalization-based optimizer) and benchmarks it against SGD and AdamW on MLP and CNN tasks. Experiment tracking uses Weights & Biases.

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
```

## Architecture

- **`muon.py`** — Core Muon optimizer. Uses Newton-Schulz iteration (`zeropower_via_newtonschulz5`) to approximate orthogonal projection of gradient matrices. Falls back to Adam for non-matrix params (biases, BatchNorm). The NS iteration uses quintic coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)` optimized for maximal slope at zero.

- **`MLP/train.py`** — Training script and CLI entry point. When using Muon, wraps it with `OptimizerGroup` that pairs Muon (for backbone Linear weights) with AdamW (for head, biases, BatchNorm params). This split is important — only 2D backbone Linear weights go through Newton-Schulz.

- **`MLP/model.py`** — Generic MLP with configurable hidden dims. Uses BatchNorm + GELU + Dropout blocks in the backbone, separate Linear head.

- **`MLP/data.py`** — Data loaders for three datasets: Forest Covertype (sklearn), Year Prediction MSD (UCI download), MNIST (torchvision). All apply StandardScaler normalization. Cached in `MLP/data_cache/`.

- **`CNN/train.py`** — CNN training script and CLI entry point. Same OptimizerGroup pattern as MLP — backbone Conv2d and Linear weights go through Newton-Schulz, head/biases/BatchNorm go to AdamW. Muon handles 4D conv weights by reshaping to 2D internally.

- **`CNN/model.py`** — Simple CNN with configurable channel widths. Backbone uses Conv2d-BN-GELU blocks with MaxPool2d, separate Linear head. AdaptiveAvgPool2d bridges backbone to head.

- **`CNN/data.py`** — CIFAR-10 data loader with standard augmentation (RandomCrop, RandomHorizontalFlip). Cached in `CNN/data_cache/`.

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
```

- **`tests/test_mlp.py`** — 38 tests covering MLP model construction, forward pass (classification + regression squeeze), data loaders (covertype, mnist, year_prediction), optimizer creation and Muon param splitting (2D backbone Linear weights only), training convergence for all 4 optimizers on both classification and regression, evaluate metrics, and SVD-based diagnostics.

- **`tests/test_cnn.py`** — 36 tests covering CNN model construction (including AdaptiveAvgPool2d variable spatial input), CIFAR-10 data loading, optimizer creation and Muon param splitting (Conv2d 4D + Linear 2D weights), 4D conv weight shape preservation after Muon steps, training convergence, evaluate metrics (classification-only, no regression keys), weight decay verification, and diagnostics (condition number, spectral norm, orthogonality error, effective rank).

### Test patterns
- Tests use small models (`channels=[8,16]`, `hidden_dims=[32,16]`) and tiny batches (B=4–8) for speed.
- `_Args` helper class mocks the argparse namespace for `create_optimizer`.
- Evaluate tests use `TensorDataset` fake loaders to avoid dataset downloads.
- Data loader tests download datasets on first run (CIFAR-10, Covertype, MNIST, Year Prediction MSD).
