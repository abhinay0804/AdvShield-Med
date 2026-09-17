"""
src/train.py
Generic training loop for AdvShield-Med.

Shared by Phase 1 (diagnostic model) and Phase 3 (gatekeeper model).
Both are standard supervised binary classification — same loop, different data.

Features:
  - Adam optimizer with optional cosine LR schedule
  - Early stopping on validation loss (patience configurable)
  - Best-val-accuracy checkpoint saving
  - Per-epoch logging to CSV (always) + wandb (when configured)

Usage:
    from src.models import build_diagnostic_model
    from src.data import get_dataloaders
    from src.train import train

    model = build_diagnostic_model()
    loaders = get_dataloaders(data_root="/path/to/chest_xray")
    trained_model = train(
        model=model,
        loaders=loaders,
        checkpoint_path="checkpoints/diagnostic_resnet18.pth",
        log_csv_path="results/phase1_log.csv",
        run_name="phase1_diagnostic",
    )
"""

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ──────────────────────────────────────────────
# Optional wandb import (fails gracefully)
# ──────────────────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    training: bool,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Run one epoch (train or eval). Returns dict with loss and accuracy."""
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress_bar = tqdm(loader, desc="Train" if training else "Val  ", leave=False, dynamic_ncols=True)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixed precision forward pass
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def _init_csv(path: str, fieldnames: list) -> None:
    """Create or overwrite CSV log file with header."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv(path: str, row: dict) -> None:
    """Append one row to an existing CSV log file."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────
def train(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    checkpoint_path: str,
    log_csv_path: str,
    run_name: str = "advshield_run",
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    use_amp: bool = True,
    use_cosine_scheduler: bool = True,
    early_stopping_patience: int = 5,
    device: torch.device = None,
    use_wandb: bool = True,
    wandb_project: str = "advshield-med",
) -> nn.Module:
    """
    Train a model with early stopping, checkpointing, and logging.

    Args:
        model: nn.Module to train (diagnostic or gatekeeper)
        loaders: dict with "train" and "val" DataLoaders
        checkpoint_path: where to save the best model weights
        log_csv_path: where to write per-epoch CSV log
        run_name: identifier for this run (used in logs and wandb)
        num_epochs: max training epochs
        learning_rate: Adam initial learning rate
        class_weights: optional tensor of class weights for CrossEntropyLoss
        use_amp: enable automatic mixed precision (AMP) for speed/memory gains
        use_cosine_scheduler: apply cosine annealing LR schedule
        early_stopping_patience: stop if val loss doesn't improve for N epochs
        device: target device (auto-detected if None)
        use_wandb: whether to log to wandb (silently skipped if wandb unavailable)
        wandb_project: wandb project name

    Returns:
        model loaded with best val-accuracy weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Disable AMP on CPU to avoid warnings/errors
    if device.type == "cpu":
        use_amp = False

    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    scheduler = None
    if use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

    # ── Logging setup ──
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "elapsed_s"]
    _init_csv(log_csv_path, fieldnames)

    if use_wandb and _WANDB_AVAILABLE:
        try:
            wandb.init(project=wandb_project, name=run_name, reinit=True)
        except Exception as e:
            print(f"wandb init failed ({e}) — continuing with CSV only")
            use_wandb = False

    # ── Training state ──
    best_val_acc = 0.0
    epochs_without_improvement = 0
    start_time = time.time()

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print("─" * 65)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = _run_epoch(model, loaders["train"], criterion, optimizer, device, training=True, scaler=scaler)
        val_metrics = _run_epoch(model, loaders["val"], criterion, None, device, training=False, scaler=scaler)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # ── Print ──
        print(
            f"{epoch:>6} "
            f"{train_metrics['loss']:>11.4f} "
            f"{train_metrics['accuracy']:>9.4f} "
            f"{val_metrics['loss']:>10.4f} "
            f"{val_metrics['accuracy']:>9.4f} "
            f"{current_lr:>10.2e}"
        )

        # ── CSV log ──
        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["accuracy"], 6),
            "lr": current_lr,
            "elapsed_s": round(elapsed, 2),
        }
        _append_csv(log_csv_path, row)

        # ── wandb log ──
        if use_wandb and _WANDB_AVAILABLE:
            try:
                wandb.log({
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "lr": current_lr,
                    "epoch": epoch,
                })
            except Exception:
                pass  # never let wandb break training

        # ── Checkpoint best model ──
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            from src.models import save_model
            save_model(model, checkpoint_path)
            print(f"         ↑ New best val acc: {best_val_acc:.4f} — checkpoint saved")
        else:
            epochs_without_improvement += 1

        # ── LR scheduler step ──
        if scheduler is not None:
            scheduler.step()

        # ── Early stopping ──
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {early_stopping_patience} epochs)")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")

    if use_wandb and _WANDB_AVAILABLE:
        try:
            wandb.finish()
        except Exception:
            pass

    # Reload best weights
    from src.models import load_model
    # We need the builder to reload — caller must use the returned model
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model
