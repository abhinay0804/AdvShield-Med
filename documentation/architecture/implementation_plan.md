# AdvShield-Med — Implementation Plan

**Full title:** AdvShield-Med: A Multi-Attack Adversarial Detection and Defense Framework for Robust Deep Learning-Based Chest X-Ray Diagnosis

---

## Locked-In Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dataset | Kermany Pediatric Pneumonia (~5.8k images) | Binary-clean, fast to iterate, matches contribution scope |
| Attack library | `torchattacks` | Minimal API, covers FGSM/PGD/DeepFool, no ART overhead |
| Backbone (both models) | ResNet-18 pretrained on ImageNet | Fast fine-tune, well-documented on small medical datasets, consistent across report |
| Runtime | Google Colab (primary), Kaggle Notebooks (backup) | Free GPU tier, quota-safe with saved checkpoints |
| Code structure | `src/` Python modules + thin Colab notebook wrappers | Clean GitHub repo, no notebook spaghetti, fast interactive iteration |
| Experiment tracking | CSV (always) + wandb (when available, wrapped in try/except) | Guaranteed local fallback, polished report figures |
| Core scope | Models 1 + 2, FGSM + PGD | Complete, defensible project |
| Training quality | Staged fine-tuning, mixed precision, enriched augmentation, class weighting | All adopted — strict improvements, no scope conflict |
| Stretch goals (in order) | Purifier → Backup Model → DeepFool → Grad-CAM | Decided at Week 9; Purifier + Backup move project to "detect and recover" narrative |

---

## Project Structure

```
advshield-med/
├── src/
│   ├── data.py           # Kermany dataset loading, enriched augmentation, splitting, class weighting
│   ├── models.py         # ResNet-18 builders + freeze_backbone/unfreeze_backbone for staged fine-tuning
│   ├── attacks.py        # FGSM/PGD/DeepFool wrappers (torchattacks), save/load tensors
│   ├── train.py          # Training loop with mixed precision, staged LR, class-weighted loss
│   ├── evaluate.py       # Accuracy, precision, recall, F1, SSIM/PSNR for purifier
│   ├── pipeline.py       # End-to-end routing: gatekeeper → purifier/backup → diagnostic
│   └── purifier.py       # Denoising autoencoder — MSE + perceptual (VGG) + SSIM loss  [Phase 5]
├── notebooks/
│   ├── phase1_baseline.ipynb        # Diagnostic CNN — staged fine-tuning
│   ├── phase2_attacks.ipynb         # Vulnerability demonstration
│   ├── phase3_gatekeeper.ipynb      # Gatekeeper training + generalization test
│   ├── phase4_pipeline.ipynb        # End-to-end integration + final results
│   ├── phase5a_purifier.ipynb       # Purifier training + evaluation  [Phase 5]
│   └── phase5b_backup.ipynb         # Adversarially-trained backup model  [Phase 5]
├── checkpoints/          # Saved model weights (.pth files, committed to Drive)
├── results/              # CSV logs, plots, final results tables
├── requirements.txt
└── README.md
```

> [!IMPORTANT]
> Save checkpoints to Google Drive at the end of every Colab session. Colab runtime disconnects will destroy any in-memory state. Use `from google.colab import drive; drive.mount('/content/drive')` at the top of every notebook.

---

## Phase 1 — Baseline Diagnostic Model

**Goal:** Train a frozen-in-place pneumonia/normal classifier that serves as the "thing we protect."

### Tasks

1. **`src/data.py`** — Kermany dataset loader (updated)
   - Download from Kaggle (`chest-xray-pneumonia` dataset) or direct URL
   - 70/15/15 stratified split (train/val/test)
   - Preprocessing: resize to 224×224, normalize with ImageNet mean/std
   - **Enriched training augmentation:**
     - Random horizontal flip + 10° rotation (existing)
     - `ColorJitter(brightness=0.2, contrast=0.2)` — simulates scanner variation
     - `GaussianBlur(kernel_size=3, p=0.1)` — low-probability blur
     - `RandomErasing(p=0.1)` — forces model off single-region reliance
   - **Class imbalance correction:** `WeightedRandomSampler` on train split, weights inversely proportional to class frequency. Corrects majority-class bias for balanced performance across both classes — not framed as directional clinical protection.
   - Return `DataLoader` objects for all three splits

2. **`src/models.py`** — Diagnostic model builder + staged fine-tuning helpers
   ```python
   def build_diagnostic_model(pretrained=True) -> nn.Module
   def freeze_backbone(model) -> nn.Module   # freeze all layers except FC head
   def unfreeze_backbone(model) -> nn.Module # release full network for Stage 2
   ```

3. **`src/train.py`** — Training loop with quality improvements
   - **Mixed precision:** `torch.cuda.amp` (`GradScaler` + `autocast`) — ~30–50% speed gain, halved GPU memory, zero accuracy cost
   - **Class-weighted loss:** `CrossEntropyLoss(weight=class_weights)` computed from training label counts
   - Adam optimizer with cosine LR schedule
   - Early stopping on val loss (patience=5)
   - Checkpoint saving: saves best val-accuracy model
   - Logs epoch/loss/acc to `results/phase1_log.csv` and wandb

4. **`notebooks/phase1_baseline.ipynb`** — **Two-stage training workflow:**

   **Stage 1 — Head only (5–10 epochs, lr=1e-3):**
   ```python
   model = freeze_backbone(model)
   model = train(model, loaders, lr=1e-3, num_epochs=10, ...)
   ```
   **Stage 2 — Full fine-tune (up to 30 epochs, lr=1e-5):**
   ```python
   model = unfreeze_backbone(model)
   model = train(model, loaders, lr=1e-5, num_epochs=30, ...)
   ```
   Produces: training curves for both stages, final test-set metrics, saved checkpoint: `checkpoints/diagnostic_resnet18.pth`

### Success Criteria
- Test accuracy ≥ 85% on clean Kermany test set (staged fine-tuning should push this higher)
- Model saved and frozen — **never modified after this point**

---

## Phase 2 — Vulnerability Demonstration

**Goal:** Show the diagnostic model breaks under attack. This is your "why this matters" evidence.

### Tasks

1. **`src/attacks.py`** — Attack wrappers
   ```python
   def run_fgsm(model, images, labels, eps) -> Tensor  # torchattacks.FGSM
   def run_pgd(model, images, labels, eps, steps, alpha) -> Tensor  # torchattacks.PGD
   ```
   - Report at 3 epsilon values: `ε ∈ {0.01, 0.03, 0.06}`
   - PGD config: steps=40, alpha=ε/10 (standard Madry et al. setting)

> [!WARNING]
> PGD with steps=40 is ~40× slower than FGSM per batch. Budget Colab GPU time accordingly — generating PGD images for the full held-out test set in Phase 3 can take 15–30 minutes on a T4. Generate and save PGD tensors to disk once; never regenerate mid-session. Use `torch.save()` / `torch.load()` for this.

2. **`notebooks/phase2_attacks.ipynb`** — Orchestrates above, produces:
   - Visual: side-by-side original vs. adversarial X-ray (human-invisible difference)
   - Results table:

| ε | Clean Acc | FGSM Acc | PGD Acc |
|---|---|---|---|
| 0.01 | — | — | — |
| 0.03 | — | — | — |
| 0.06 | — | — | — |

   - Saved: perturbed image tensors for Phase 3 training dataset construction

### Success Criteria
- Visible accuracy drops under attack (expected: FGSM drops to ~30–60% depending on ε, PGD lower)
- Phase 2 results table is a standalone midterm deliverable

---

## Phase 3 — Gatekeeper Training & Generalization Test

**Goal:** Train a binary clean/adversarial detector, then test if it generalizes to PGD it never saw.

### Gatekeeper Dataset Construction

| Split | Source | Label |
|---|---|---|
| Train (gatekeeper) | Train split clean images | 0 (clean) |
| Train (gatekeeper) | FGSM-attacked versions of same images (ε=0.03) | 1 (adversarial) |
| Val (gatekeeper) | Val split clean + FGSM (same construction) | 0/1 |
| **Test — SEEN** | Test split clean + FGSM | 0/1 |
| **Test — UNSEEN** | Test split clean + **PGD**-attacked (held out, never in training) | 0/1 |

> [!IMPORTANT]
> PGD-attacked images must be generated **before** training starts and held out completely. The gatekeeper must never see PGD examples during training or validation — only at final test time. This is the generalization test.

### Tasks

1. **`src/models.py`** — Gatekeeper model builder
   ```python
   def build_gatekeeper_model(num_classes=2, pretrained=True) -> nn.Module:
       # Same ResNet-18 architecture, different final weights
       # Can optionally use a smaller/shallower head — binary task is simpler
   ```

2. **`src/train.py`** — Reuse generic training loop for gatekeeper
   - Binary cross-entropy loss
   - Logs to `results/phase3_log.csv`

3. **`src/evaluate.py`** — Detection-specific metrics
   ```python
   def evaluate_detection(model, dataloader) -> dict:
       # Returns: accuracy, precision, recall, F1, confusion matrix
   ```

4. **`notebooks/phase3_gatekeeper.ipynb`** — Produces:
   - Loss/accuracy curves
   - Detection results table:

| Test Set | Precision | Recall | F1 |
|---|---|---|---|
| Clean images | — | — | — |
| FGSM (seen attack) | — | — | — |
| PGD (unseen — generalization) | — | — | — |

   - Saved checkpoint: `checkpoints/gatekeeper_resnet18.pth`

### Success Criteria
- FGSM detection F1 ≥ 0.85 (seen attack — should be high)
- PGD generalization result: report honestly regardless of value
  - High F1 → strong claim: gatekeeper generalizes across attack types
  - Low F1 → valid finding: demonstrates real-world generalization gap in adversarial defenses

---

## Phase 4 — Full Pipeline Integration

**Goal:** Wire Model 1 + Model 2 together and prove the end-to-end system works.

### Pipeline Logic (`src/pipeline.py`)

```python
def run_pipeline(gatekeeper, diagnostic_model, image, threshold=0.5):
    """
    Returns: (decision, confidence, diagnosis_or_none)
    - If gatekeeper predicts 'adversarial' with prob > threshold → REJECT
    - If gatekeeper predicts 'clean' → pass to diagnostic_model → return diagnosis
    """
```

> [!IMPORTANT]
> **Threshold tuning is required — do not ship with hardcoded 0.5.** In Phase 4, run a threshold sweep over `{0.3, 0.4, 0.5, 0.6, 0.7}` on the validation set and plot the clean-pass-rate vs. adversarial-detection-rate tradeoff (a simple 2-line curve). Choose the threshold that maximises detection without over-rejecting clean images. Report the chosen value and its justification in the results. Panels notice when a threshold looks arbitrary.

### Tasks

1. **`src/pipeline.py`** — End-to-end routing logic (see above)

2. **`notebooks/phase4_pipeline.ipynb`** — Integration test + final results, produces:

**Final Results Table:**

| Metric | Value |
|---|---|
| Diagnostic model accuracy — clean images | (from Phase 1) |
| Diagnostic model accuracy — under FGSM (undefended) | (from Phase 2) |
| Diagnostic model accuracy — under PGD (undefended) | (from Phase 2) |
| Gatekeeper detection F1 — FGSM (seen) | (from Phase 3) |
| Gatekeeper detection F1 — PGD (unseen) | (from Phase 3) |
| End-to-end accuracy — clean images through pipeline | (should ≈ Phase 1 baseline) |
| End-to-end accuracy — FGSM images (post-gatekeeper filtering) | (should be >> undefended) |
| End-to-end accuracy — PGD images (post-gatekeeper filtering) | (depends on gatekeeper generalization) |

### Success Criteria
- Clean images pass through unimpeded, maintaining baseline diagnostic accuracy
- Adversarial images get flagged and blocked before reaching diagnostic model
- End-to-end accuracy on attacked images demonstrably higher than undefended baseline

---

## Phase 5 — Correction Layer (decide at Week 9, build in priority order)

> [!NOTE]
> Do not start Phase 5 until Phases 1–4 are fully complete and verified. **Both P1 and P2 together move the contribution from "detect and reject" to "detect and recover"** — a materially stronger narrative. If only one is achievable, P1 (Purifier) is the higher-novelty item. P3 and P4 are independent additions that do not depend on each other.

### Priority 1: Purifier — Image-Level Correction (`src/purifier.py`)
**What it does:** A denoising autoencoder that reconstructs a clean image from an adversarially-perturbed input, which is then re-sent to the *original, frozen* diagnostic model. The diagnostic model remains untouched — deployment-realism argument fully intact.

**Architecture:** Convolutional encoder–decoder (4–5 blocks), skip connections for detail preservation, 224×224 → 224×224.

**Loss function — mandatory combined loss:**
```python
loss = λ1 * MSE(output, clean)              # pixel-level fidelity
     + λ2 * perceptual_loss(output, clean)  # VGG feature-space comparison
     + λ3 * (1 - SSIM(output, clean))       # structural/diagnostic detail
```
> [!IMPORTANT]
> Pure MSE loss produces blurry reconstructions (autoencoders average over plausible pixels). This directly threatens diagnostic utility. Perceptual + SSIM components specifically preserve the structural detail a radiologist — or diagnostic model — needs. Combined loss is **not optional** if the purifier is built.

**Training data:** Clean images (target) paired with FGSM/PGD-attacked versions (input) — already generated in Phase 2/3.

**Evaluation:**
- Reconstruction quality: PSNR and SSIM on purified vs. original clean images
- Diagnostic recovery: `purified image → frozen diagnostic model → accuracy` — the compound metric that matters
- Compare: purified accuracy vs. raw attacked accuracy vs. clean baseline

**Notebook:** `notebooks/phase5a_purifier.ipynb`  
**Checkpoint:** `checkpoints/purifier.pth`

**Real-time feasibility:** Purifier inference = one forward pass (~15–40ms on GPU). Full pipeline latency (gatekeeper + purifier + diagnostic) ≈ 100ms per image — low enough not to be a bottleneck in a research prototype context. Report measured latency as a metric; avoid "clinical deployment" language, which invites panel questions about PACS integration, batch throughput, and regulatory pathways that are explicitly out of scope.

### Priority 2: Adversarially-Trained Backup — Decision-Level Correction
**What it does:** A secondary ResNet-18 trained on mixed clean + adversarial batches (adversarial training). Used as an alternative fallback path for gatekeeper-flagged images alongside the purifier — both paths are evaluated independently.

**Training:** Same loop as Phase 1 (`src/train.py`), but each minibatch includes 50% clean + 50% FGSM-attacked images with original labels. Mixed precision (`torch.cuda.amp`) is inherited automatically via the shared training loop — no separate implementation needed. The model learns to be robust to perturbations at training time.

**Evaluation:** Accuracy-under-attack on FGSM (seen) and PGD (unseen) — direct comparison to the undefended diagnostic model from Phase 2.

**Report:** Both purifier path and backup path results in a side-by-side comparison table. Report honestly — one underperforming the other is a valid finding and demonstrates the tradeoff between image-level vs. decision-level correction.

**Notebook:** `notebooks/phase5b_backup.ipynb`  
**Checkpoint:** `checkpoints/backup_resnet18.pth`

### Priority 3: DeepFool Attack
- Library: `torchattacks.DeepFool` (same library, minimal extra setup)
- Add to Phase 2 vulnerability table: DeepFool accuracy drop
- Add to Phase 3 generalization table: gatekeeper vs. DeepFool (second unseen attack type)
- Strengthens novelty claim: generalization tested against two unseen attack types, not one

### Priority 4: Grad-CAM Visualizations
- Library: `pytorch-grad-cam` (install only when starting this phase)
- **Target layer for ResNet-18:** `model.layer4[-1]` — last residual block before global average pool
- Show: what the diagnostic model attends to on clean vs. adversarial images
- Show: what the gatekeeper attends to when flagging (likely different regions — this contrast is the interesting result)
- Show: what the purifier's output looks like overlaid with attention (if purifier is built)
- Output: heatmap figures for report, side-by-side with original X-ray

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Colab GPU quota exhausted | Mount Drive checkpoints every session; Kaggle Notebooks as backup |
| Gatekeeper doesn't generalize to PGD | Valid honest finding — report it, don't hide it |
| Dataset too slow / large | Kermany is already the small choice — no fallback needed |
| Phase 3 runs long (most likely) | Trim Phase 5 to P1 (Purifier) only if needed — still a complete "detect and recover" story |
| Purifier MSE-only reconstruction is blurry | Combined loss (MSE + perceptual + SSIM) is mandatory — do not use MSE alone |
| Purifier recovers image quality but not diagnostic accuracy | Compound evaluation exposes this — report it, it's a valid finding about the limits of image-level correction |
| wandb API issues | CSV logging is always-on fallback; wandb wrapped in try/except |

---

## Explicit Non-Scope (do not add mid-project)

- Hospital portal / web dashboard UI
- Audit logging / admin notification
- Detector robustness against adaptive attacks
- Federated learning
- Black-box transfer attacks
- Dataset upgrade to NIH ChestX-ray14

These belong in the report's "Future Work" section, not in the codebase.

---

## Verification Plan

### Automated
- `pytest` unit tests for `src/evaluate.py` metric computations (spot-check against sklearn)
- Sanity check: run pipeline on 10 known-clean and 10 known-adversarial images, verify routing

### Manual
- Visual inspection: adversarial images look identical to clean to human eye (proves attack is valid)
- Loss curves: verify no divergence, gatekeeper doesn't massively overfit
- Results tables: all four tables (Phase 1, 2, 3, 4) filled with real numbers before report submission

---

## `requirements.txt`

```
torch>=2.0
torchvision>=0.15
torchattacks>=3.4
wandb
scikit-learn
matplotlib
numpy
pandas
Pillow
tqdm
torchmetrics>=1.0   # SSIM/PSNR for purifier evaluation

# --- Phase 5 stretch goals only — install when starting Phase 5 ---
# pytorch-grad-cam>=1.4
```
