import argparse
import sys
import os

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

# Add parent directory so we can import muon.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from muon import MuonJordan, MuonLLM
from CNN.model import CNN
from CNN.data import get_cifar10_loaders
from common.logger import setup_logger
from common.metrics import compute_weight_diagnostics, compute_gradient_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="CNN Optimizer Benchmark")
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10",
                        help="Dataset to use")
    parser.add_argument("--optim", choices=["sgd", "adamw", "muon-jordan", "muon-llm"],
                        default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N training steps")
    parser.add_argument("--channels", type=int, nargs="+",
                        default=[32, 64, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_diagnostics", action="store_true",
                        help="Log detailed optimizer diagnostics to W&B at each eval step")
    return parser.parse_args()


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class OptimizerGroup:
    """Wraps multiple optimizers with a unified interface."""
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


def create_optimizer(model, args):
    if args.optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr,
                               momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    elif args.optim in ("muon-jordan", "muon-llm"):
        # Backbone Conv2d and Linear weights get Newton-Schulz; everything else gets AdamW
        modules = dict(model.named_modules())
        muon_params = []
        for name, p in model.named_parameters():
            if 'backbone' in name and name.endswith('.weight'):
                mod_name = name.rsplit('.', 1)[0]
                if isinstance(modules[mod_name], (nn.Conv2d, nn.Linear)):
                    muon_params.append(p)

        muon_ids = {id(p) for p in muon_params}
        adam_params = [p for p in model.parameters() if id(p) not in muon_ids]

        MuonCls = MuonJordan if args.optim == "muon-jordan" else MuonLLM

        return OptimizerGroup(
            MuonCls(muon_params, lr=args.lr, momentum=0.95, steps=5,
                    weight_decay=args.weight_decay, nesterov=True),
            torch.optim.AdamW(adam_params, lr=args.lr,
                              weight_decay=args.weight_decay),
        )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    correct = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        total_loss += criterion(out, y).item() * X.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        n_samples += X.size(0)

    model.train()
    return {
        "eval/loss": total_loss / n_samples,
        "eval/accuracy": correct / n_samples,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device(args.device)

    log = setup_logger("cnn", args.dataset, args.optim)
    log.info(f"Using device: {device}")

    # Load data
    train_loader, test_loader, in_channels, num_classes = \
        get_cifar10_loaders(args.batch_size, args.seed)

    # Create model
    model = CNN(in_channels, num_classes, args.channels, args.dropout).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {param_count:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args)

    # W&B
    wandb.init(
        project="muon",
        name=f"cnn-{args.dataset}-{args.optim}",
        config=vars(args),
    )

    # Step-0 evaluation
    global_step = 0
    metrics = evaluate(model, test_loader, criterion, device)
    if args.log_diagnostics:
        weight_diag = compute_weight_diagnostics(model)
        metrics.update(weight_diag)
        for k, v in list(metrics.items()):
            if k.endswith("/sv_histogram"):
                metrics[k] = wandb.Histogram(v)
    wandb.log(metrics, step=global_step)
    summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()
                        if not k.startswith("diagnostics/"))
    log.info(f"[Step {global_step}] {summary}")

    # Training loop
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            wandb.log({"train/loss": loss.item()}, step=global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % args.eval_every == 0:
                if args.log_diagnostics:
                    grad_diag = compute_gradient_diagnostics(model)
                    weight_diag = compute_weight_diagnostics(model)
                metrics = evaluate(model, test_loader, criterion, device)
                if args.log_diagnostics:
                    metrics.update(grad_diag)
                    metrics.update(weight_diag)
                    for k, v in list(metrics.items()):
                        if k.endswith("/sv_histogram"):
                            metrics[k] = wandb.Histogram(v)
                wandb.log(metrics, step=global_step)
                summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()
                                    if not k.startswith("diagnostics/"))
                log.info(f"[Step {global_step}] {summary}")

    # Final evaluation
    if args.log_diagnostics:
        grad_diag = compute_gradient_diagnostics(model)
        weight_diag = compute_weight_diagnostics(model)
    metrics = evaluate(model, test_loader, criterion, device)
    if args.log_diagnostics:
        metrics.update(grad_diag)
        metrics.update(weight_diag)
        for k, v in list(metrics.items()):
            if k.endswith("/sv_histogram"):
                metrics[k] = wandb.Histogram(v)
    wandb.log(metrics, step=global_step)
    summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()
                        if not k.startswith("diagnostics/"))
    log.info(f"Final eval: {summary}")

    wandb.finish()


if __name__ == "__main__":
    main()
