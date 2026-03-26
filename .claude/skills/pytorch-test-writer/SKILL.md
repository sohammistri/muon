---
name: pytorch-test-writer
description: >
  Write high-quality, prioritized pytest test suites for PyTorch deep learning code — models, optimizers,
  data loaders, training loops, and diagnostics. Use this skill whenever the user asks to write tests,
  add test coverage, create a test file, or verify correctness for any PyTorch / deep learning code in this
  repository. Also trigger when the user says things like "test this model", "add tests for the CNN",
  "make sure the optimizer works", "verify the training loop", or "check the data pipeline". Even if the
  user just says "test it" or "write tests" in the context of DL code, use this skill.
---

# PyTorch Deep Learning Test Writer

You are writing tests for a PyTorch deep learning codebase. Your job is to produce a comprehensive,
prioritized test suite that catches real bugs — not just tests that pass. Every test you write should
have a reason to exist: it either guards against a specific failure mode, validates a critical invariant,
or ensures a contract between components holds.

## Process

Follow these steps in order. Do not skip the exploration phase — understanding the code is what separates
useful tests from checkbox tests.

### Step 1: Explore the Code Under Test

Before writing a single test, read and understand the code you're testing. This means:

1. **Read every source file** in the module being tested (model, data, train, optimizer, metrics)
2. **Trace the data flow**: input shapes → model forward → loss → backward → optimizer step → evaluation
3. **Identify the parameter split logic**: which parameters go to Muon (Newton-Schulz) vs AdamW? This is
   a common source of subtle bugs — a Conv2d weight accidentally routed to AdamW won't crash but will
   silently degrade training
4. **Note shape transformations**: Muon reshapes 4D conv weights to 2D for Newton-Schulz, then reshapes
   back. If this goes wrong, training silently produces garbage
5. **Check evaluation function signatures**: MLP evaluate takes a `task` argument, CNN evaluate does not.
   Tests must match the actual API

The goal is to build a mental model of what can go wrong. Tests follow from that understanding.

### Step 2: Classify Test Priority

Every test gets a priority level. This isn't decoration — it determines whether a test is worth writing
at all and how robust it needs to be.

**CRITICAL** — Tests that catch bugs which would silently corrupt training or produce wrong results
without raising errors. These are non-negotiable:
- Model forward pass produces correct output shapes (wrong shape = crash or silent broadcasting bugs)
- Optimizer parameter split correctness (wrong split = Muon applied to BatchNorm = silent degradation)
- Weight shape preservation after optimizer step (Muon reshape bug = silent garbage)
- Loss decreases on a tiny overfitting batch (training loop fundamentally broken = wasted GPU hours)
- Gradients flow through all parameters (dead params = silent capacity waste)
- Evaluation function returns correct metric keys and value ranges

**HIGH** — Tests that validate important contracts and catch common mistakes:
- Data loader batch shapes, dtypes, and label ranges
- Model construction with default and custom configs
- Weight decay actually reduces weight norms
- Evaluation restores training mode after eval
- Parameter count scales with model config (sanity check for architecture bugs)
- Diagnostics functions return expected keys and prefixes

**OPTIONAL** — Nice-to-have tests for edge cases and extra confidence:
- Variable spatial input sizes (AdaptiveAvgPool2d flexibility)
- Normalization constants are reasonable
- Backbone contains expected layer types
- Regression output squeeze behavior (output_dim=1)

### Step 3: Write the Tests

Follow these conventions — they exist because this project already has 74 tests that follow them, and
consistency matters more than personal preference.

#### File Structure and Conventions

```python
"""Tests for {MODULE} model, data loading, optimizer creation, and training."""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import from the module under test
from {MODULE}.model import {ModelClass}
from {MODULE}.data import {data_loaders}
from {MODULE}.train import create_optimizer, evaluate, OptimizerGroup
from muon import MuonJordan, MuonLLM
from common.metrics import compute_weight_diagnostics, compute_gradient_diagnostics
```

#### Test Organization

Group tests into classes by concern. Use this exact ordering — it matches the dependency chain
(you need a model before you can test an optimizer, you need an optimizer before you can test training):

1. `TestModel` — Construction and forward pass
2. `TestData` — Data loaders, shapes, dtypes, label ranges
3. `TestOptimizerCreation` — Optimizer types, parameter splitting
4. `TestOptimizerGroup` — The unified optimizer wrapper
5. `TestTrainingStep` — Forward + backward + optimizer step convergence
6. `TestEvaluate` — Evaluation function outputs
7. `TestDiagnostics` — SVD-based weight and gradient diagnostics

Use `# ---------------------------------------------------------------------------` separators between
classes for readability.

#### The `_Args` Helper

The `create_optimizer` function expects an argparse namespace. Mock it with a simple class:

```python
class _Args:
    """Minimal args namespace for create_optimizer."""
    def __init__(self, optim, lr=1e-3, weight_decay=1e-4):
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
```

#### Model Sizes for Tests

Always use small models and tiny batches. Tests must run in seconds, not minutes:
- MLP: `hidden_dims=[32, 16]`, batch size 4-8
- CNN: `channels=[8, 16]`, batch size 2-4, spatial 32x32
- Never use default model sizes in tests — they're too large

#### Key Test Patterns

**Loss convergence test** (CRITICAL): Train for 5 steps on a tiny fixed batch. Loss must decrease.
Parametrize across all 4 optimizers (`sgd`, `adamw`, `muon-jordan`, `muon-llm`):

```python
@pytest.mark.parametrize("optim_name", ["sgd", "adamw", "muon-jordan", "muon-llm"])
def test_loss_decreases(self, setup, optim_name):
    model, criterion, x, y = setup
    optimizer = create_optimizer(model, _Args(optim_name, lr=1e-2))
    model.train()
    initial_loss = None
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial_loss
```

**Parameter split test** (CRITICAL): Verify Muon only receives the weights it should. For MLP, that's
2D backbone Linear weights. For CNN, that's 2D Linear + 4D Conv2d backbone weights. Check:
- `muon_param_count + adam_param_count == total_param_count`
- All Muon params have expected ndim (2 for MLP, 2 or 4 for CNN)

**Shape preservation test** (CRITICAL): After a Muon optimizer step, verify Conv2d weights are still 4D
and Linear weights are still 2D. The Newton-Schulz iteration operates on 2D — if the reshape-back fails,
the model breaks silently.

**Evaluation test**: Use `TensorDataset` with random data to avoid dataset downloads:
```python
dataset = torch.utils.data.TensorDataset(
    torch.randn(32, input_dim), torch.randint(0, num_classes, (32,))
)
loader = torch.utils.data.DataLoader(dataset, batch_size=16)
```

**Gradient flow test**: After one forward + backward pass, every named parameter must have a non-None,
non-zero gradient.

**Diagnostics test**: Verify `compute_weight_diagnostics` and `compute_gradient_diagnostics` return
dicts with expected key patterns (`condition_number`, `spectral_norm`, `orthogonality_error`,
`effective_rank`, `grad_norm`). Check the `diagnostics/` prefix.

#### What NOT to Test

- Don't test PyTorch internals (nn.Linear computes matrix multiply correctly)
- Don't test exact numerical values that depend on random seeds (fragile)
- Don't test the W&B integration or file I/O in unit tests
- Don't test `parse_args` or `main()` — those are integration concerns
- Don't test that `torch.compile` works — that's PyTorch's problem

#### Module-Specific Differences

Pay attention to these differences between MLP and CNN code:

| Aspect | MLP | CNN |
|--------|-----|-----|
| `evaluate()` signature | `(model, loader, criterion, task, device)` | `(model, loader, criterion, device)` |
| Muon-eligible params | 2D backbone `nn.Linear` weights only | 2D `nn.Linear` + 4D `nn.Conv2d` backbone weights |
| Task types | classification + regression | classification only |
| Eval metrics | accuracy OR (mse + r2) | accuracy only |
| Forward squeeze | `output_dim=1` squeezes last dim | No squeeze |

Getting these wrong is a common mistake when copying test patterns between modules.

### Step 4: Add Priority Markers as Comments

At the top of each test class or before groups of tests, add a comment indicating the priority level:

```python
# [CRITICAL] These tests catch silent training corruption
class TestTrainingStep:
    ...

# [HIGH] These tests validate data pipeline contracts
class TestData:
    ...
```

This helps future developers understand which tests matter most during triage.

### Step 5: Save the Tests

Save the test file to `tests/test_{module_name}.py` where `{module_name}` matches the directory name
in lowercase (e.g., `tests/test_mlp.py`, `tests/test_cnn.py`).

After writing, run `pytest tests/test_{module_name}.py -v` to verify all tests pass. Fix any failures
before presenting the result.

### Step 6: Report Summary

After writing and verifying tests, present a summary table:

```
| Priority | Count | Categories |
|----------|-------|------------|
| CRITICAL |   N   | forward shape, param split, convergence, ... |
| HIGH     |   N   | data shapes, construction, weight decay, ... |
| OPTIONAL |   N   | edge cases, layer types, ... |
```

This gives the user a quick picture of coverage and where the test effort was spent.
