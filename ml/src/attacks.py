"""
src/attacks.py
Adversarial attack wrappers for AdvShield-Med.

Wraps torchattacks FGSM and PGD with:
  - Consistent interface across attack types
  - Batch-by-batch generation with progress reporting
  - Save/load to disk (important for PGD — never regenerate mid-session)
  - Returns attacked image tensors in the same normalized space as input

IMPORTANT: PGD with steps=40 is ~40x slower than FGSM per batch.
           Always generate and save PGD tensors once, then reload.
           See save_attacked_tensors() / load_attacked_tensors() below.

Usage:
    from src.attacks import generate_attacked_dataset, save_attacked_tensors, load_attacked_tensors

    # Generate and save (do this once per attack type + epsilon)
    adv_images, labels = generate_attacked_dataset(
        model=diagnostic_model,
        loader=test_loader,
        attack_type="pgd",
        eps=0.03,
        device=device,
    )
    save_attacked_tensors(adv_images, labels, "results/pgd_test_eps0.03.pt")

    # Reload in later sessions
    adv_images, labels = load_attacked_tensors("results/pgd_test_eps0.03.pt")
"""

from pathlib import Path
from typing import Literal, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import torchattacks


# ──────────────────────────────────────────────
# Attack factory
# ──────────────────────────────────────────────
def _build_attack(
    model: nn.Module,
    attack_type: Literal["fgsm", "pgd", "deepfool"],
    eps: float,
    pgd_steps: int = 40,
    pgd_alpha: float = None,
) -> torchattacks.Attack:
    """
    Instantiate a torchattacks attack object.

    Args:
        model: the victim model (must be in eval mode)
        attack_type: "fgsm" or "pgd"
        eps: maximum L-infinity perturbation (e.g. 0.03)
        pgd_steps: number of PGD iterations (default 40, Madry et al.)
        pgd_alpha: PGD step size (default eps/10 if None)

    Returns:
        torchattacks.Attack instance
    """
    attack_type = attack_type.lower()

    if attack_type == "fgsm":
        return torchattacks.FGSM(model, eps=eps)

    elif attack_type == "pgd":
        alpha = pgd_alpha if pgd_alpha is not None else eps / 10
        return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=pgd_steps)
        
    elif attack_type == "deepfool":
        return torchattacks.DeepFool(model, steps=50, overshoot=0.02)

    else:
        raise ValueError(f"Unknown attack_type: '{attack_type}'. Choose 'fgsm', 'pgd', or 'deepfool'.")


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────
def generate_attacked_dataset(
    model: nn.Module,
    loader: DataLoader,
    attack_type: Literal["fgsm", "pgd", "deepfool"],
    eps: float,
    device: torch.device = None,
    pgd_steps: int = 40,
    pgd_alpha: float = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate adversarial examples for every batch in a DataLoader.

    The diagnostic model must be in eval mode with frozen weights.
    Attacked images remain in the same normalized pixel space as the input.

    Args:
        model: victim model (diagnostic CNN, frozen)
        loader: DataLoader yielding (clean_images, labels)
        attack_type: "fgsm", "pgd", or "deepfool"
        eps: perturbation budget
        device: target device (auto-detected if None)
        pgd_steps: PGD iteration count (ignored for FGSM)
        pgd_alpha: PGD step size (defaults to eps/10)

    Returns:
        (adv_images, labels) as CPU tensors, concatenated across all batches
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    attack = _build_attack(model, attack_type, eps, pgd_steps, pgd_alpha)
    attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    all_adv_images = []
    all_labels = []

    desc = f"Generating {attack_type.upper()} (ε={eps})"
    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device)
        labels_dev = labels.to(device)

        with torch.enable_grad():
            adv_images = attack(images, labels_dev)

        all_adv_images.append(adv_images.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_adv_images, dim=0), torch.cat(all_labels, dim=0)


# ──────────────────────────────────────────────
# Save / Load (critical for PGD)
# ──────────────────────────────────────────────
def save_attacked_tensors(
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    path: str,
) -> None:
    """
    Save generated adversarial images and labels to a .pt file.

    Always do this immediately after generation — especially for PGD,
    which takes 15-30 minutes on a Colab T4 GPU for the full test set.

    Args:
        adv_images: (N, C, H, W) tensor of adversarial images
        labels: (N,) tensor of ground truth labels
        path: output file path (e.g. "results/pgd_test_eps0.03.pt")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": adv_images, "labels": labels}, path)
    print(f"Adversarial tensors saved → {path}  ({adv_images.shape[0]} images)")


def load_attacked_tensors(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load previously saved adversarial images and labels.

    Args:
        path: path to .pt file created by save_attacked_tensors()

    Returns:
        (adv_images, labels) tuple of tensors
    """
    data = torch.load(path, map_location="cpu")
    adv_images = data["images"]
    labels = data["labels"]
    print(f"Adversarial tensors loaded ← {path}  ({adv_images.shape[0]} images)")
    return adv_images, labels


def make_dataloader_from_tensors(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
) -> DataLoader:
    """
    Wrap image and label tensors into a DataLoader for evaluation.

    Args:
        images: (N, C, H, W) tensor
        labels: (N,) tensor
        batch_size: mini-batch size

    Returns:
        DataLoader (shuffle=False — deterministic order for evaluation)
    """
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
