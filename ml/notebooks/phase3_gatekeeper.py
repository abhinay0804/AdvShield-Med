import os, sys
import torch
import numpy as np

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

from src.data    import get_dataloaders
from src.models  import (build_diagnostic_model, build_gatekeeper_model,
                          freeze_model, load_model, save_model, count_parameters)
from src.attacks import (generate_attacked_dataset, save_attacked_tensors,
                          load_attacked_tensors, make_dataloader_from_tensors)
from src.train   import train
from src.evaluate import evaluate_model, print_report, build_results_row

from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIAG_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'diagnostic_resnet18.pth')
GATE_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'gatekeeper_resnet18.pth')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'results')
LOG_CSV         = os.path.join(RESULTS_DIR, 'phase3_log.csv')
BATCH_SIZE      = 32
TRAINING_EPS    = 0.03   # epsilon used to generate FGSM training data for gatekeeper

print(f'Device: {device}')
print(f'Training epsilon: {TRAINING_EPS}')








loaders, _ = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=0)
train_loader = loaders['train']
val_loader   = loaders['val']
test_loader  = loaders['test']








diag_model = load_model(build_diagnostic_model, DIAG_CHECKPOINT, device=device)
diag_model = freeze_model(diag_model)
print('Diagnostic model loaded and frozen.')








class GatekeeperDataLoaderWrapper:
    def __init__(self, base_loader, diag_model, eps, device):
        self.base_loader = base_loader
        self.diag_model = diag_model
        self.eps = eps
        self.device = device
        import torchattacks
        self.attack = torchattacks.FGSM(diag_model, eps=eps)
        self.attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __iter__(self):
        import torch
        for images, labels in self.base_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.enable_grad():
                adv_images = self.attack(images, labels)
            
            clean_labels = torch.zeros(images.size(0), dtype=torch.long, device=self.device)
            adv_labels = torch.ones(adv_images.size(0), dtype=torch.long, device=self.device)
            
            # Combine clean and adversarial images
            all_images = torch.cat([images, adv_images], dim=0)
            all_labels = torch.cat([clean_labels, adv_labels], dim=0)
            
            # Shuffle them so every batch has a 50/50 mix!
            perm = torch.randperm(all_images.size(0), device=self.device)
            all_images = all_images[perm]
            all_labels = all_labels[perm]
            
            # Split back into two batches to keep batch_size consistent
            half = images.size(0)
            yield all_images[:half], all_labels[:half]
            yield all_images[half:], all_labels[half:]
            
    def __len__(self):
        return len(self.base_loader) * 2

gate_train_loader = GatekeeperDataLoaderWrapper(train_loader, diag_model, TRAINING_EPS, device)
gate_val_loader = GatekeeperDataLoaderWrapper(val_loader, diag_model, TRAINING_EPS, device)
print("Gatekeeper data loaders initialized (on-the-fly generation with batch shuffling).")



pgd_path = os.path.join(RESULTS_DIR, f'pgd_test_eps{TRAINING_EPS}.pt')
pgd_adv_images, pgd_true_labels = load_attacked_tensors(pgd_path)

# For gatekeeper evaluation: all these are adversarial → label=1
pgd_gate_labels = torch.ones(len(pgd_adv_images), dtype=torch.long)
pgd_gate_loader = make_dataloader_from_tensors(pgd_adv_images, pgd_gate_labels, BATCH_SIZE)

# Also build a clean test loader for gatekeeper eval (label=0)
clean_test_images_list = [imgs for imgs, _ in test_loader]
clean_test_images = torch.cat(clean_test_images_list, dim=0)
clean_gate_labels = torch.zeros(len(clean_test_images), dtype=torch.long)
clean_gate_loader = make_dataloader_from_tensors(clean_test_images, clean_gate_labels, BATCH_SIZE)

print(f'PGD held-out test: {len(pgd_adv_images)} images')
print(f'Clean test (for gatekeeper eval): {len(clean_test_images)} images')








print('Skipping training loop and loading best checkpoint...')
gatekeeper = build_gatekeeper_model()
gatekeeper = gatekeeper.to(device)
gatekeeper.load_state_dict(torch.load(GATE_CHECKPOINT))



import os
os.makedirs('training_plots', exist_ok=True)
import os
os.makedirs('training_plots', exist_ok=True)
import os
os.makedirs('training_plots', exist_ok=True)
import os
os.makedirs('training_plots', exist_ok=True)
import matplotlib.pyplot as plt

log = pd.read_csv(LOG_CSV)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(log['epoch'], log['train_loss'], label='Train')
axes[0].plot(log['epoch'], log['val_loss'],   label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves — Gatekeeper'); axes[0].legend()

axes[1].plot(log['epoch'], log['train_acc'], label='Train')
axes[1].plot(log['epoch'], log['val_acc'],   label='Val')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curves — Gatekeeper'); axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join('training_plots', 'phase3_curves.png'), dpi=150)









# Also need FGSM test set (seen attack, for comparison)
fgsm_test_path = os.path.join(RESULTS_DIR, f'fgsm_test_eps{TRAINING_EPS}.pt')
if not os.path.exists(fgsm_test_path):
    print('Generating FGSM test set for gatekeeper eval...')
    fgsm_test_imgs, fgsm_test_lbls = generate_attacked_dataset(
        diag_model, test_loader, 'fgsm', eps=TRAINING_EPS, device=device
    )
    save_attacked_tensors(fgsm_test_imgs, fgsm_test_lbls, fgsm_test_path)
else:
    fgsm_test_imgs, _ = load_attacked_tensors(fgsm_test_path)

fgsm_gate_labels = torch.ones(len(fgsm_test_imgs), dtype=torch.long)
fgsm_gate_loader = make_dataloader_from_tensors(fgsm_test_imgs, fgsm_gate_labels, BATCH_SIZE)








conditions = [
    ('Clean images (no attack)',   clean_gate_loader),
    ('FGSM ε=0.03 (seen attack)',  fgsm_gate_loader),
    ('PGD ε=0.03 (unseen — generalization test)', pgd_gate_loader),
]

results_rows = []
for label, loader in conditions:
    metrics = evaluate_model(gatekeeper, loader, device=device)
    print_report(metrics, title=f'Gatekeeper — {label}')
    results_rows.append(build_results_row(label, metrics))

df = pd.DataFrame(results_rows)
print('\n', df.to_string(index=False))
df.to_csv(os.path.join(RESULTS_DIR, 'phase3_gatekeeper_results.csv'), index=False)

# Success check for seen attack
fgsm_metrics = evaluate_model(gatekeeper, fgsm_gate_loader, device=device)
if fgsm_metrics['f1'] >= 0.85:
    print('\n✓ SUCCESS: FGSM (seen) detection F1 ≥ 0.85')
else:
    print(f'\n⚠ WARNING: FGSM detection F1 = {fgsm_metrics["f1"]:.4f} < 0.85 target')







