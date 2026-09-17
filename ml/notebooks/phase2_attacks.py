import os, sys

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = '/content/drive/MyDrive/advshield-med'
    DATA_ROOT    = '/content/drive/MyDrive/chest_xray'
    sys.path.insert(0, PROJECT_ROOT)
else:
    PROJECT_ROOT = os.path.abspath('.')
    DATA_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'chest_xray')
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.data    import get_dataloaders
from src.models  import build_diagnostic_model, freeze_model, load_model
from src.attacks import generate_attacked_dataset, save_attacked_tensors, make_dataloader_from_tensors
from src.evaluate import evaluate_model, print_report, build_results_row

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT  = os.path.join(PROJECT_ROOT, 'checkpoints', 'diagnostic_resnet18.pth')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
BATCH_SIZE  = 32

print(f'Device: {device}')

model = load_model(build_diagnostic_model, CHECKPOINT, device=device)
model = freeze_model(model)
print('Diagnostic model loaded and frozen.')

loaders, _ = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=0)
test_loader = loaders['test']

clean_metrics = evaluate_model(model, test_loader, device=device)
print_report(clean_metrics, title='Diagnostic Model — Clean')

import pandas as pd

EPSILONS = [0.01, 0.03, 0.06]
results_rows = [build_results_row('Clean (no attack)', clean_metrics)]

# Store PGD tensors at eps=0.03 for the held-out generalization test in Phase 3
HELD_OUT_EPS = 0.03
held_out_pgd_saved = False

for eps in EPSILONS:
    print(f'\n{'='*60}')
    print(f'  ε = {eps}')
    print(f'{'='*60}')

    # ── FGSM ────────────────────────────────────────────────────────────────
    fgsm_imgs, fgsm_labels = generate_attacked_dataset(
        model, test_loader, 'fgsm', eps=eps, device=device
    )
    fgsm_loader  = make_dataloader_from_tensors(fgsm_imgs, fgsm_labels, BATCH_SIZE)
    fgsm_metrics = evaluate_model(model, fgsm_loader, device=device)
    print_report(fgsm_metrics, title=f'FGSM ε={eps}')
    results_rows.append(build_results_row(f'FGSM ε={eps}', fgsm_metrics))

    # ── PGD ─────────────────────────────────────────────────────────────────
    pgd_imgs, pgd_labels = generate_attacked_dataset(
        model, test_loader, 'pgd', eps=eps, device=device
    )
    # Save PGD tensors immediately
    pgd_path = os.path.join(RESULTS_DIR, f'pgd_test_eps{eps}.pt')
    save_attacked_tensors(pgd_imgs, pgd_labels, pgd_path)

    pgd_loader  = make_dataloader_from_tensors(pgd_imgs, pgd_labels, BATCH_SIZE)
    pgd_metrics = evaluate_model(model, pgd_loader, device=device)
    print_report(pgd_metrics, title=f'PGD ε={eps}')
    results_rows.append(build_results_row(f'PGD ε={eps}', pgd_metrics))

print('\n✓ All attacks generated and saved.')

df = pd.DataFrame(results_rows)
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULTS_DIR, 'phase2_attack_results.csv'), index=False)
print('\nSaved → results/phase2_attack_results.csv')

import matplotlib.pyplot as plt
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def denormalize(t):
    """Undo ImageNet normalisation for display."""
    img = t.cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)

# Take the first 3 test images
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images[:3]

# Generate FGSM at eps=0.03 for visualization only
from src.attacks import _build_attack
import torch.nn.functional as F
attack = _build_attack(model, 'fgsm', eps=0.03)
attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
adv_samples = attack(sample_images.to(device), sample_labels[:3].to(device)).cpu()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i in range(3):
    axes[0, i].imshow(denormalize(sample_images[i]))
    axes[0, i].set_title(f'Clean (label={sample_labels[i].item()})')
    axes[0, i].axis('off')

    axes[1, i].imshow(denormalize(adv_samples[i]))
    axes[1, i].set_title(f'FGSM ε=0.03')
    axes[1, i].axis('off')

fig.suptitle('Clean vs Adversarial — perturbation is human-invisible', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'phase2_clean_vs_adv.png'), dpi=150)
plt.show()
