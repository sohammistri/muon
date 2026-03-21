import torch
import torch.nn as nn


def _compute_sv_metrics(W):
    """Compute singular-value-based metrics for a 2D weight tensor."""
    W = W.float().cpu()
    S = torch.linalg.svdvals(W)
    S_pos = S[S > 1e-7]

    if len(S_pos) == 0:
        return {
            "condition_number": float("inf"),
            "effective_rank": 0.0,
            "spectral_norm": 0.0,
            "sv_entropy": 0.0,
            "sv_histogram": [],
        }

    condition_number = (S_pos[0] / S_pos[-1]).item()
    spectral_norm = S_pos[0].item()

    # Normalized SVs as probability distribution
    p = S_pos / S_pos.sum()
    sv_entropy = -(p * p.log()).sum().item()
    effective_rank = torch.exp(torch.tensor(sv_entropy)).item()

    return {
        "condition_number": condition_number,
        "effective_rank": effective_rank,
        "spectral_norm": spectral_norm,
        "sv_entropy": sv_entropy,
        "sv_histogram": S.tolist(),
    }


def _compute_orthogonality_error(W):
    """Compute how far W is from having orthogonal columns/rows.

    Returns normalized Frobenius distance: ||G/||G||_F - I||_F / sqrt(k)
    where G = W^TW or WW^T (whichever is smaller) and k = min(m, n).
    """
    W = W.float().cpu()
    m, n = W.shape
    if m >= n:
        G = W.mT @ W  # n x n
        k = n
    else:
        G = W @ W.mT  # m x m
        k = m

    G_normalized = G / (G.norm() + 1e-7)
    I = torch.eye(k, device=W.device)
    return (G_normalized - I).norm().item() / (k ** 0.5)


@torch.no_grad()
def compute_weight_diagnostics(model, layer_filter=None):
    """Compute per-layer weight matrix diagnostics (SVD-based).

    Args:
        model: The nn.Module to analyze.
        layer_filter: Optional callable(name, module) -> bool.
                      Defaults to all nn.Linear layers.

    Returns:
        Dict with keys like "diagnostics/{layer_name}/{metric_name}".
        sv_histogram values are raw lists (caller wraps with wandb.Histogram).
    """
    if layer_filter is None:
        layer_filter = lambda name, mod: isinstance(mod, nn.Linear)

    metrics = {}
    for name, module in model.named_modules():
        if not layer_filter(name, module):
            continue

        W = module.weight.data
        if W.ndim < 2:
            continue

        sv_metrics = _compute_sv_metrics(W)
        orth_error = _compute_orthogonality_error(W)

        prefix = f"diagnostics/{name}"
        for k, v in sv_metrics.items():
            metrics[f"{prefix}/{k}"] = v
        metrics[f"{prefix}/orthogonality_error"] = orth_error
        metrics[f"{prefix}/weight_frobenius_norm"] = W.float().norm().item()

    return metrics


@torch.no_grad()
def compute_gradient_diagnostics(model, layer_filter=None):
    """Compute per-layer gradient diagnostics. Call before optimizer.zero_grad().

    Args:
        model: The nn.Module to analyze.
        layer_filter: Optional callable(name, module) -> bool.
                      Defaults to all nn.Linear layers.

    Returns:
        Dict with keys like "diagnostics/{layer_name}/{metric_name}".
    """
    if layer_filter is None:
        layer_filter = lambda name, mod: isinstance(mod, nn.Linear)

    metrics = {}
    for name, module in model.named_modules():
        if not layer_filter(name, module):
            continue

        W = module.weight
        if W.grad is None:
            continue

        prefix = f"diagnostics/{name}"
        grad_norm = W.grad.float().norm().item()
        weight_norm = W.data.float().norm().item()

        metrics[f"{prefix}/grad_norm"] = grad_norm
        if weight_norm > 1e-7:
            metrics[f"{prefix}/update_to_weight_ratio"] = grad_norm / weight_norm

    return metrics
