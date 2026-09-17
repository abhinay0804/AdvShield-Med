# Sprint 1: Training Quality Upgrades (Phase 1)

## Goal
Implement several best-practice training quality improvements to our Phase 1 codebase before running any models. This ensures our diagnostic backbone is as strong as possible, and avoids majority-class bias.

## Scope of Changes

1. **Staged Fine-Tuning (`src/models.py`)**
   - Added `freeze_backbone()` to lock all layers except the FC head.
   - Added `unfreeze_backbone()` to release the network for full fine-tuning.
   - Allows high learning rate training for the new head (Stage 1), then low learning rate fine-tuning for the whole network (Stage 2).

2. **Mixed Precision Training (`src/train.py`)**
   - Integrated `torch.cuda.amp.autocast` and `GradScaler`.
   - Dramatically reduces memory usage (allowing larger batches if needed) and speeds up training on modern GPUs without accuracy loss.

3. **Class Imbalance Correction (`src/data.py` & `src/train.py`)**
   - Computed inverse-frequency class weights for the Kermany dataset (which leans heavily toward Pneumonia).
   - Applied a `WeightedRandomSampler` to the training dataloader so each batch sees a balanced 50/50 mix.
   - Note: The clinical framing for this is purely to ensure "balanced performance across both classes" rather than prioritizing one direction over the other, as over-diagnosis is just as much a failure mode as under-diagnosis.

4. **Enriched Data Augmentations (`src/data.py`)**
   - Added `ColorJitter(brightness=0.2, contrast=0.2)` to simulate natural scanner variation.
   - Added `GaussianBlur(kernel_size=3, p=0.1)` and `RandomErasing(p=0.1)` to force the model to learn robust, distributed features rather than memorizing small regions of the images.

## Next Steps
Once these changes are coded into the `src/` modules, we will update `notebooks/phase1_baseline.ipynb` to execute the staged workflow, then run the Phase 1 training to obtain our frozen diagnostic model.
