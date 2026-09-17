# AdvShield-Med — Task Tracker

## Phase 1 — Baseline Diagnostic Model
- [x] Scaffold project directory (`advshield-med/`)
- [x] `requirements.txt` (updated: +torchmetrics)
- [x] `src/data.py` — Kermany loader, stratified split, enriched augmentation (ColorJitter, GaussianBlur, RandomErasing), WeightedRandomSampler for class balance
- [x] `src/models.py` — `build_diagnostic_model()`, `build_gatekeeper_model()`, `freeze_backbone()`, `unfreeze_backbone()`
- [x] `src/train.py` — mixed precision (amp), class-weighted loss, early stopping, CSV+wandb logging
- [x] `src/evaluate.py` — accuracy, precision, recall, F1, confusion matrix
- [x] `notebooks/phase1_baseline.ipynb` — two-stage training: freeze_backbone (lr=1e-3) → unfreeze_backbone (lr=1e-5)
- [x] **UPDATE notebook:** reflect staged fine-tuning workflow, mixed precision, WeightedRandomSampler
- [ ] **RUN Stage 1:** freeze backbone, train head only (10 epochs, lr=1e-3)
- [ ] **RUN Stage 2:** unfreeze, fine-tune full network (30 epochs, lr=1e-5)
- [ ] **VERIFY:** test accuracy ≥ 85%; model frozen, 0 trainable params

## Phase 2 — Vulnerability Demonstration
- [x] `src/attacks.py` — `run_fgsm()` and `run_pgd()` wrappers (torchattacks)
- [x] `notebooks/phase2_attacks.ipynb` — sweep ε ∈ {0.01, 0.03, 0.06}
- [ ] **RUN:** generate + **save to disk** PGD-attacked test images (held-out; never used in gatekeeper training)
- [ ] **RUN:** produce clean/FGSM/PGD accuracy table (midterm deliverable)
- [ ] **VERIFY:** visible accuracy drop under attack

## Phase 3 — Gatekeeper Training & Generalization Test
- [ ] **RUN:** build clean+FGSM training dataset (label 0/1)
- [ ] **RUN:** confirm held-out PGD test set loaded (never seen during training)
- [x] `src/models.py` — `build_gatekeeper_model()` ResNet-18 builder
- [x] `notebooks/phase3_gatekeeper.ipynb` — train gatekeeper, save `checkpoints/gatekeeper_resnet18.pth`
- [ ] **RUN:** evaluate on: clean, FGSM (seen), PGD (unseen) — report precision/recall/F1
- [ ] **VERIFY:** FGSM detection F1 ≥ 0.85; PGD reported honestly regardless

## Phase 4 — Full Pipeline Integration
- [x] `src/pipeline.py` — `run_pipeline()` gatekeeper→diagnostic routing
- [x] `notebooks/phase4_pipeline.ipynb` — threshold sweep + integration test
- [ ] **RUN:** threshold sweep {0.3, 0.4, 0.5, 0.6, 0.7} on validation set, select and justify
- [ ] **RUN:** produce final 9-row results table
- [ ] **RUN:** inline metric sanity checks vs sklearn
- [ ] **VERIFY:** clean images pass through unimpeded; adversarial images flagged before diagnostic model

## Phase 5 — Correction Layer (after Phase 4 verified, Week 9 decision)

### P1 — Purifier (image-level correction)
- [ ] `src/purifier.py` — convolutional encoder-decoder, skip connections, 224×224
- [ ] Combined loss: MSE + perceptual (VGG feature space) + SSIM — **mandatory, not optional**
- [ ] `notebooks/phase5a_purifier.ipynb` — train on FGSM/PGD → clean pairs
- [ ] **RUN:** evaluate reconstruction quality (PSNR, SSIM)
- [ ] **RUN:** evaluate compound metric: purified image → frozen diagnostic → accuracy
- [ ] **RUN:** measure + report end-to-end inference latency (~100ms target)
- [ ] Save `checkpoints/purifier.pth`

### P2 — Adversarially-Trained Backup (decision-level correction)
- [ ] `notebooks/phase5b_backup.ipynb` — train ResNet-18 on 50% clean + 50% FGSM batches
- [ ] **RUN:** evaluate accuracy-under-attack: FGSM (seen) + PGD (unseen)
- [ ] **RUN:** side-by-side comparison table: purifier path vs backup path vs undefended
- [ ] Save `checkpoints/backup_resnet18.pth`

### P3 — DeepFool Attack
- [ ] Add `run_deepfool()` to `src/attacks.py`
- [ ] **RUN:** add DeepFool to Phase 2 vulnerability table
- [ ] **RUN:** add DeepFool to Phase 3 generalization test (second unseen attack type)

### P4 — Grad-CAM Visualizations
- [ ] Install `pytorch-grad-cam` (only when starting this phase)
- [ ] Target layer: `model.layer4[-1]` for ResNet-18
- [ ] **RUN:** heatmaps for diagnostic model (clean vs. adversarial)
- [ ] **RUN:** heatmaps for gatekeeper (what it attends to when flagging)
- [ ] **RUN:** if purifier built — heatmaps on purified output

## Repo Hygiene
- [x] `README.md` with project description, setup instructions, phase structure
- [x] `src/__init__.py` (makes src/ importable as package)
- [ ] Google Drive checkpoint sync confirmed working (verify on first Colab session)
