# Muon Optimizer

Implementation of the **Muon optimizer** — a Newton-Schulz orthogonalization-based optimizer — benchmarked against SGD and AdamW on MLP and CNN tasks.

Muon applies a quintic Newton-Schulz iteration to approximate the orthogonal projection of gradient matrices, encouraging weight matrices to stay near the orthogonal manifold during training. Non-matrix parameters (biases, BatchNorm) fall back to Adam updates.

## How It Works

The core idea: instead of updating weights with raw gradients, Muon orthogonalizes the gradient via Newton-Schulz iteration before applying it. The iteration uses optimized quintic coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)` that maximize the slope at zero for fast convergence.

Two variants are implemented:

- **Muon-Jordan** — scales the update by `sqrt(max(1, rows/cols))`, following Jordan's formulation
- **Muon-LLM** — scales by `0.2 * sqrt(max(rows, cols))`, tuned for large-scale language model training

Both use Nesterov momentum with `beta=0.95` and 5 Newton-Schulz steps by default. For Conv2d layers, 4D weight tensors are reshaped to 2D before orthogonalization and restored afterward.

## Project Structure

```
muon.py              # Core optimizer: Newton-Schulz iteration + MuonJordan/MuonLLM classes
MLP/
  train.py           # MLP training CLI with OptimizerGroup (Muon + AdamW)
  model.py           # Configurable MLP: BatchNorm + GELU + Dropout blocks
  data.py            # Loaders for Covertype, Year Prediction MSD, MNIST
CNN/
  train.py           # CNN training CLI with same OptimizerGroup pattern
  model.py           # Simple CNN: Conv2d-BN-GELU blocks + AdaptiveAvgPool2d
  data.py            # CIFAR-10 loader with standard augmentation
common/
  metrics.py         # SVD-based weight diagnostics (condition number, effective rank, etc.)
  logger.py          # Colored console + file logging
tests/
  test_mlp.py        # 38 tests covering MLP model, data, optimizers, training
  test_cnn.py        # 36 tests covering CNN model, data, optimizers, training
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### MLP Benchmarks

```bash
# Train with Muon-Jordan on MNIST
uv run python MLP/train.py --optim muon-jordan --dataset mnist --epochs 10

# Train with AdamW on Covertype
uv run python MLP/train.py --optim adamw --dataset covertype --epochs 10

# Regression task on Year Prediction MSD
uv run python MLP/train.py --optim muon-llm --dataset year_prediction --epochs 10

# Enable SVD diagnostics (logged to W&B)
uv run python MLP/train.py --optim muon-jordan --dataset mnist --log_diagnostics
```

**MLP options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--optim` | `adamw` | `sgd`, `adamw`, `muon-jordan`, `muon-llm` |
| `--dataset` | inferred | `covertype`, `year_prediction`, `mnist` |
| `--lr` | `1e-3` | Learning rate |
| `--epochs` | `10` | Number of epochs |
| `--batch_size` | `1024` | Batch size |
| `--hidden_dims` | `512 256 256 128` | Hidden layer dimensions |
| `--dropout` | `0.1` | Dropout rate |
| `--weight_decay` | `1e-4` | Weight decay |
| `--log_diagnostics` | off | Log SVD metrics to W&B |

### CNN Benchmarks

```bash
# Train with Muon-Jordan on CIFAR-10
uv run python CNN/train.py --optim muon-jordan --dataset cifar10 --epochs 20

# Enable diagnostics
uv run python CNN/train.py --optim muon-llm --dataset cifar10 --log_diagnostics
```

**CNN options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--optim` | `adamw` | `sgd`, `adamw`, `muon-jordan`, `muon-llm` |
| `--dataset` | `cifar10` | `cifar10` |
| `--lr` | `1e-3` | Learning rate |
| `--epochs` | `10` | Number of epochs |
| `--batch_size` | `128` | Batch size |
| `--channels` | `32 64 128` | Conv channel widths |
| `--log_diagnostics` | off | Log SVD metrics to W&B |

## Diagnostics

When `--log_diagnostics` is enabled, the following SVD-based metrics are computed for each weight matrix and logged to Weights & Biases:

- **Condition number** — ratio of largest to smallest singular value
- **Effective rank** — exponential of singular value entropy
- **Spectral norm** — largest singular value
- **Orthogonality error** — Frobenius distance from the nearest orthogonal matrix
- **Gradient norms** — per-layer gradient magnitude tracking

These metrics are key for understanding Muon's effect on weight geometry compared to standard optimizers.

## Testing

```bash
# Run all tests (74 total)
uv run python -m pytest tests/ -v

# MLP tests only (38 tests)
uv run python -m pytest tests/test_mlp.py -v

# CNN tests only (36 tests)
uv run python -m pytest tests/test_cnn.py -v
```

Tests cover model construction, forward passes, data loading, optimizer parameter splitting, training convergence, evaluation metrics, and SVD diagnostics. They use small models and tiny batches for speed.

## Experiment Tracking

All runs are logged to [Weights & Biases](https://wandb.ai). Set your W&B API key before running:

```bash
wandb login
```

Logs are also saved locally to `logs/`.
