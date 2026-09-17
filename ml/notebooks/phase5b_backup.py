import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath('.')
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'chest_xray')
sys.path.insert(0, PROJECT_ROOT)

from src.data     import get_dataloaders
from src.models   import build_diagnostic_model, freeze_model, load_model
from src.train    import train
from src.evaluate import evaluate_model, print_report
from src.attacks  import load_attacked_tensors, make_dataloader_from_tensors
import torchattacks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 32
TRAINING_EPS = 0.03

# 1. Setup Data Loaders
loaders, _ = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=0)
diag_model_for_attack = load_model(build_diagnostic_model, os.path.join(PROJECT_ROOT, 'checkpoints', 'diagnostic_resnet18.pth'), device)
diag_model_for_attack = freeze_model(diag_model_for_attack)

# Wrapper to generate 50% clean / 50% FGSM batches
# But preserving DISEASE labels (unlike Gatekeeper which swaps labels to 0/1)
class BackupDataLoaderWrapper:
    def __init__(self, base_loader, victim_model, eps, device):
        self.base_loader = base_loader
        self.device = device
        self.attack = torchattacks.FGSM(victim_model, eps=eps)
        self.attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __iter__(self):
        for images, labels in self.base_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.enable_grad():
                adv_images = self.attack(images, labels)
            
            # Combine clean and adversarial
            all_images = torch.cat([images, adv_images], dim=0)
            # The disease label remains exactly the same for both clean and adv
            all_labels = torch.cat([labels, labels], dim=0)
            
            # Shuffle the 2x size batch
            perm = torch.randperm(all_images.size(0), device=self.device)
            all_images = all_images[perm]
            all_labels = all_labels[perm]
            
            # Yield in two standard-sized batches
            half = images.size(0)
            yield all_images[:half], all_labels[:half]
            yield all_images[half:], all_labels[half:]
            
    def __len__(self):
        return len(self.base_loader) * 2

train_loader = BackupDataLoaderWrapper(loaders['train'], diag_model_for_attack, TRAINING_EPS, device)
val_loader   = BackupDataLoaderWrapper(loaders['val'], diag_model_for_attack, TRAINING_EPS, device)

# 2. Build and Train Backup Model
backup_model = build_diagnostic_model().to(device)

backup_checkpoint = os.path.join(PROJECT_ROOT, 'checkpoints', 'backup_resnet18.pth')
log_csv = os.path.join(PROJECT_ROOT, 'results', 'phase5b_backup_log.csv')

print("\n--- Training Backup Model (Adversarial Training) ---")
backup_model = train(
    model=backup_model,
    loaders={'train': train_loader, 'val': val_loader},
    checkpoint_path=backup_checkpoint,
    log_csv_path=log_csv,
    run_name="phase5_backup",
    early_stopping_patience=5,
    num_epochs=15
)

# 3. Evaluate Backup Model
print("\n--- Evaluating Backup Model ---")

fgsm_test_imgs, fgsm_test_labels = load_attacked_tensors(os.path.join(PROJECT_ROOT, 'results', f'fgsm_test_eps{TRAINING_EPS}.pt'))
pgd_test_imgs, pgd_test_labels = load_attacked_tensors(os.path.join(PROJECT_ROOT, 'results', f'pgd_test_eps{TRAINING_EPS}.pt'))

fgsm_loader = make_dataloader_from_tensors(fgsm_test_imgs, fgsm_test_labels, BATCH_SIZE)
pgd_loader = make_dataloader_from_tensors(pgd_test_imgs, pgd_test_labels, BATCH_SIZE)

metrics_fgsm = evaluate_model(backup_model, fgsm_loader, device)
print_report(metrics_fgsm, title='Backup Model (FGSM Seen)')

metrics_pgd = evaluate_model(backup_model, pgd_loader, device)
print_report(metrics_pgd, title='Backup Model (PGD Unseen)')
