"""
src/pipeline.py
End-to-end AdvShield-Med pipeline (Phase 4).

Routes an input image through the gatekeeper first, then to the diagnostic
model only if the gatekeeper judges the image as clean.

This is the core operational output of the project:
  - Clean image → gatekeeper passes it → diagnostic CNN → diagnosis
  - Adversarial image → gatekeeper flags it → REJECT (no diagnosis returned)

Threshold tuning (IMPORTANT):
  Do NOT use the default threshold=0.5 in your final results.
  Run sweep_threshold() on your validation set first, choose the threshold
  that maximises adversarial detection without over-rejecting clean images,
  and report the chosen value in your results table.

Usage:
    from src.pipeline import AdvShieldPipeline

    pipeline = AdvShieldPipeline(
        gatekeeper=gatekeeper_model,
        diagnostic=diagnostic_model,
        threshold=0.62,   # tuned on val set — not hardcoded 0.5
        device=device,
    )

    result = pipeline.classify(image_tensor)
    # result = {"decision": "clean", "diagnosis": 1, "gate_confidence": 0.12}
    #       or {"decision": "adversarial", "diagnosis": None, "gate_confidence": 0.91}
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class AdvShieldPipeline:
    """
    Gatekeeper → Diagnostic two-stage pipeline.

    Args:
        gatekeeper: trained gatekeeper CNN (binary: 0=clean, 1=adversarial)
        diagnostic: trained diagnostic CNN (binary: 0=normal, 1=pneumonia) — FROZEN
        threshold: adversarial confidence threshold for rejection (tune this!)
        device: target device
    """

    def __init__(
        self,
        gatekeeper: nn.Module,
        diagnostic: nn.Module,
        purifier: Optional[nn.Module] = None,
        threshold: float = 0.5,
        device: torch.device = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.threshold = threshold

        self.gatekeeper = gatekeeper.to(device).eval()
        self.diagnostic = diagnostic.to(device).eval()
        self.purifier = purifier.to(device).eval() if purifier is not None else None

        # Verify diagnostic model is frozen
        trainable = sum(p.numel() for p in self.diagnostic.parameters() if p.requires_grad)
        if trainable > 0:
            raise RuntimeError(
                f"Diagnostic model has {trainable} trainable parameters — "
                "call freeze_model(diagnostic) before creating pipeline."
            )

    # ──────────────────────────────────────────────
    # Single-image inference
    # ──────────────────────────────────────────────
    def classify(self, image: torch.Tensor) -> Dict:
        """
        Run the full pipeline on a single image tensor.

        Args:
            image: (C, H, W) or (1, C, H, W) tensor, ImageNet-normalized

        Returns:
            dict with:
              decision        — "clean" or "adversarial"
              gate_confidence — probability that gatekeeper assigns to 'adversarial' class
              diagnosis       — 0 (normal) or 1 (pneumonia), or None if rejected
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # add batch dim
        image = image.to(self.device)

        with torch.no_grad():
            # Gatekeeper inference
            gate_logits = self.gatekeeper(image)
            gate_probs = F.softmax(gate_logits, dim=1)
            adv_confidence = gate_probs[0, 1].item()  # prob of class 1 = adversarial
            decision = "adversarial" if adv_confidence >= self.threshold else "clean"

            if self.purifier is not None:
                # SANITIZE-FIRST LOGIC
                purified_image = self.purifier(image)
                diag_logits = self.diagnostic(purified_image)
                diagnosis = diag_logits.argmax(dim=1).item()
                
                return {
                    "decision": decision,
                    "gate_confidence": adv_confidence,
                    "diagnosis": diagnosis,
                }
            else:
                # TRADITIONAL GATEKEEPER ROUTING LOGIC
                if adv_confidence >= self.threshold:
                    return {
                        "decision": "adversarial",
                        "gate_confidence": adv_confidence,
                        "diagnosis": None,
                    }

                diag_logits = self.diagnostic(image)
                diagnosis = diag_logits.argmax(dim=1).item()

                return {
                    "decision": "clean",
                    "gate_confidence": adv_confidence,
                    "diagnosis": diagnosis,
                }

    # ──────────────────────────────────────────────
    # Batch evaluation
    # ──────────────────────────────────────────────
    def evaluate_batch(
        self,
        loader: DataLoader,
        true_image_type: str,
    ) -> Dict:
        """
        Evaluate the pipeline over a DataLoader.

        Args:
            loader: DataLoader yielding (images, labels)
                    - labels are ground-truth diagnoses (0/1)
                    - images may be clean or adversarial
            true_image_type: "clean" or "adversarial" — what these images actually are
                             (used to compute gate accuracy separately from diag accuracy)

        Returns:
            dict with pipeline performance metrics
        """
        total = 0
        gate_correct = 0      # gatekeeper routing decision is correct
        diag_correct = 0      # diagnostic prediction is correct (for passed images)
        passed_count = 0      # images passed through to diagnostic
        rejected_count = 0    # images rejected by gatekeeper
        all_gate_confs = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Pipeline eval ({true_image_type})"):
                images = images.to(self.device)
                labels = labels.numpy()

                gate_logits = self.gatekeeper(images)
                gate_probs = F.softmax(gate_logits, dim=1)
                adv_confs = gate_probs[:, 1].cpu().numpy()
                all_gate_confs.extend(adv_confs)

                for i, (adv_conf, label) in enumerate(zip(adv_confs, labels)):
                    total += 1
                    rejected = adv_conf >= self.threshold

                    # Gate routing correctness (analytical only if using purifier)
                    if true_image_type == "adversarial" and rejected:
                        gate_correct += 1
                    elif true_image_type == "clean" and not rejected:
                        gate_correct += 1
                        
                    if self.purifier is not None:
                        # Sanitize-first bypasses rejection logic
                        passed_count += 1
                        purified_img = self.purifier(images[i].unsqueeze(0))
                        diag_out = self.diagnostic(purified_img)
                        pred = diag_out.argmax(dim=1).item()
                        if pred == label:
                            diag_correct += 1
                        # Track rejected strictly for logging/analytical purposes
                        if rejected:
                            rejected_count += 1
                    else:
                        # Traditional routing logic
                        if not rejected:
                            passed_count += 1
                            # Diagnostic inference for this image
                            diag_out = self.diagnostic(images[i].unsqueeze(0))
                            pred = diag_out.argmax(dim=1).item()
                            if pred == label:
                                diag_correct += 1
                        else:
                            rejected_count += 1

        results = {
            "total_images":        total,
            "gate_accuracy":       gate_correct / total,
            "passed_to_diagnostic": passed_count,
            "rejected":            rejected_count,
            "rejection_rate":      rejected_count / total,
            "diag_accuracy_on_passed": diag_correct / passed_count if passed_count > 0 else None,
            "mean_gate_confidence": float(np.mean(all_gate_confs)),
        }
        return results

    # ──────────────────────────────────────────────
    # Threshold sweep (run this before finalising threshold)
    # ──────────────────────────────────────────────
    def sweep_threshold(
        self,
        clean_loader: DataLoader,
        adversarial_loader: DataLoader,
        thresholds: List[float] = None,
    ) -> List[Dict]:
        """
        Sweep over candidate thresholds and report tradeoff metrics.

        Run this on the VALIDATION set (not test set) to select threshold.
        Report the chosen threshold and this table in your results.

        Args:
            clean_loader: DataLoader of clean val images
            adversarial_loader: DataLoader of adversarial val images
            thresholds: list of threshold values to evaluate
                        (default: [0.3, 0.4, 0.5, 0.6, 0.7])

        Returns:
            list of dicts, one per threshold, with clean_pass_rate and adv_detect_rate
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        # Collect gate confidences for all val images
        def get_confs(loader: DataLoader) -> np.ndarray:
            confs = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(self.device)
                    logits = self.gatekeeper(images)
                    probs = F.softmax(logits, dim=1)
                    confs.extend(probs[:, 1].cpu().numpy())
            return np.array(confs)

        print("Collecting gatekeeper confidences on val set...")
        clean_confs = get_confs(clean_loader)
        adv_confs = get_confs(adversarial_loader)

        rows = []
        print(f"\n{'Threshold':>10} {'Clean Pass Rate':>16} {'Adv Detect Rate':>16}")
        print("─" * 46)

        for t in thresholds:
            clean_pass_rate = (clean_confs < t).mean()       # clean correctly passed
            adv_detect_rate = (adv_confs >= t).mean()         # adversarial correctly rejected

            print(f"{t:>10.2f} {clean_pass_rate:>15.4f}  {adv_detect_rate:>15.4f}")
            rows.append({
                "threshold":       t,
                "clean_pass_rate": round(float(clean_pass_rate), 4),
                "adv_detect_rate": round(float(adv_detect_rate), 4),
            })

        print("\nChoose the threshold that maximises adv_detect_rate without "
              "unacceptably reducing clean_pass_rate.")
        return rows
