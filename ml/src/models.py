"""
src/models.py
ResNet-18 model builders for AdvShield-Med.

Two model types:
  1. Diagnostic model  — binary classifier (Normal / Pneumonia)
  2. Gatekeeper model  — binary detector (Clean / Adversarial)

Both share the same ResNet-18 backbone pretrained on ImageNet.
The final fully-connected layer is replaced for binary output.
Models are kept architecturally identical to keep the report clean
(backbone choice is not an experimental variable in this project).

Usage:
    from src.models import build_diagnostic_model, build_gatekeeper_model, load_model

    diag = build_diagnostic_model(pretrained=True)
    gate = build_gatekeeper_model(pretrained=True)

    # After training, save:
    torch.save(diag.state_dict(), "checkpoints/diagnostic_resnet18.pth")

    # To reload:
    diag = load_model(build_diagnostic_model, "checkpoints/diagnostic_resnet18.pth")
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# ──────────────────────────────────────────────
# Internal builder
# ──────────────────────────────────────────────
def _build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Instantiate ResNet-18 with ImageNet pretrained weights, replacing the
    final FC layer for `num_classes` outputs.

    Args:
        num_classes: output dimension (2 for both our binary tasks)
        pretrained: load ImageNet weights (recommended: True for transfer learning)

    Returns:
        nn.Module ready for fine-tuning
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the classification head
    in_features = model.fc.in_features          # 512 for ResNet-18
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ──────────────────────────────────────────────
# Public builders
# ──────────────────────────────────────────────
def build_diagnostic_model(pretrained: bool = True) -> nn.Module:
    """
    Build the diagnostic CNN (Model 1).

    Task: Binary classification — Normal (0) vs. Pneumonia (1).
    This model is trained once in Phase 1 and then FROZEN.
    Do not pass this model's optimizer to any subsequent training loop.

    Args:
        pretrained: use ImageNet pretrained weights (default True)

    Returns:
        ResNet-18 with 2-class output head
    """
    model = _build_resnet18(num_classes=2, pretrained=pretrained)
    return model


def build_gatekeeper_model(pretrained: bool = True) -> nn.Module:
    """
    Build the gatekeeper CNN (Model 2).

    Task: Binary detection — Clean (0) vs. Adversarial (1).
    Same architecture as the diagnostic model; weights are trained separately.

    Args:
        pretrained: use ImageNet pretrained weights (default True)

    Returns:
        ResNet-18 with 2-class output head
    """
    model = _build_resnet18(num_classes=2, pretrained=pretrained)
    return model


# ──────────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────────
def save_model(model: nn.Module, path: str) -> None:
    """
    Save model state dict to disk.

    Args:
        model: trained nn.Module
        path: file path (e.g. "checkpoints/diagnostic_resnet18.pth")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def load_model(
    builder_fn,
    path: str,
    device: torch.device = None,
    pretrained: bool = False,
) -> nn.Module:
    """
    Load a saved model state dict into a fresh model instance.

    Args:
        builder_fn: one of build_diagnostic_model or build_gatekeeper_model
        path: path to the .pth checkpoint file
        device: target device (defaults to CUDA if available, else CPU)
        pretrained: whether to init backbone with ImageNet weights before
                    loading checkpoint (usually False — checkpoint has trained weights)

    Returns:
        nn.Module in eval() mode, moved to device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = builder_fn(pretrained=pretrained)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded ← {path}  (device: {device})")
    return model


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters of a model (requires_grad = False).

    Use this on the diagnostic model after Phase 1 to ensure it is
    never accidentally modified during gatekeeper training.

    Args:
        model: nn.Module to freeze

    Returns:
        The same model with all parameters frozen
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("Model frozen — all parameters set to requires_grad=False")
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Freeze all layers of ResNet-18 except the final fully connected head.
    Used for Stage 1 fine-tuning.
    """
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("Backbone frozen — only FC head is trainable.")
    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """
    Release all layers for full network fine-tuning.
    Used for Stage 2 fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone unfrozen — all parameters are trainable.")
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
