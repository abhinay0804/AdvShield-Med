import os, sys, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import math

PROJECT_ROOT = os.path.abspath('.')
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'chest_xray')
sys.path.insert(0, PROJECT_ROOT)

from src.data     import get_dataloaders
from src.models   import build_diagnostic_model, freeze_model, load_model
from src.purifier import PurifierCNN, CombinedLoss, ssim
from src.evaluate import evaluate_model, print_report
import torchattacks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 8
ACCUM_STEPS = 4
TRAINING_EPS = 0.03
EPOCHS = 10

# 1. Setup Dataloaders and Diagnostic Model
loaders, _ = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=0)
diag_model = load_model(build_diagnostic_model, os.path.join(PROJECT_ROOT, 'checkpoints', 'diagnostic_resnet18.pth'), device)
diag_model = freeze_model(diag_model)

# 2. On-the-fly Dataset Wrapper for Autoencoder
class AutoencoderDataLoaderWrapper:
    def __init__(self, base_loader, diag_model, eps, device):
        self.base_loader = base_loader
        self.device = device
        self.attack = torchattacks.FGSM(diag_model, eps=eps)
        self.attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __iter__(self):
        for clean_images, _ in self.base_loader:
            clean_images = clean_images.to(self.device)
            # We need the true labels to generate the attack
            # Wait, our base loader provides them
            break
            
# Let's fix the Wrapper to use the true labels
class AutoencoderDataLoaderWrapper:
    def __init__(self, base_loader, diag_model, eps, device):
        self.base_loader = base_loader
        self.device = device
        self.attack = torchattacks.FGSM(diag_model, eps=eps)
        self.attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __iter__(self):
        for clean_images, labels in self.base_loader:
            clean_images = clean_images.to(self.device)
            labels = labels.to(self.device)
            with torch.enable_grad():
                adv_images = self.attack(clean_images, labels)
            # The Purifier is trained to map (adv_images) -> (clean_images)
            yield adv_images, clean_images
            
    def __len__(self):
        return len(self.base_loader)

train_ae_loader = AutoencoderDataLoaderWrapper(loaders['train'], diag_model, TRAINING_EPS, device)
val_ae_loader   = AutoencoderDataLoaderWrapper(loaders['val'], diag_model, TRAINING_EPS, device)

# 3. Initialize Model and Loss
purifier = PurifierCNN().to(device)
criterion = CombinedLoss(mse_weight=1.0, ssim_weight=0.5, perceptual_weight=0.1, device=device)
optimizer = optim.Adam(purifier.parameters(), lr=1e-4)

# 4. Training Loop
best_val_loss = float('inf')
save_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'purifier.pth')
os.makedirs(os.path.dirname(save_path), exist_ok=True)

print("\n--- Training Purifier (MSE + SSIM + Perceptual VGG) ---")
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10}")
print("-" * 35)

for epoch in range(1, EPOCHS + 1):
    purifier.train()
    train_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(train_ae_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
    for i, (adv_imgs, clean_imgs) in enumerate(pbar):
        purified_imgs = purifier(adv_imgs)
        loss, l_mse, l_ssim, l_perc = criterion(purified_imgs, clean_imgs)
        
        # Scale loss by accumulation steps
        loss = loss / ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_ae_loader):
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss += (loss.item() * ACCUM_STEPS)
        pbar.set_postfix({'loss': f"{loss.item() * ACCUM_STEPS:.4f}"})
        
    train_loss /= len(train_ae_loader)
    
    purifier.eval()
    val_loss = 0.0
    with torch.no_grad():
        for adv_imgs, clean_imgs in val_ae_loader:
            purified_imgs = purifier(adv_imgs)
            loss, _, _, _ = criterion(purified_imgs, clean_imgs)
            val_loss += loss.item()
    val_loss /= len(val_ae_loader)
    
    print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(purifier.state_dict(), save_path)

print("Training complete. Best weights saved.")
purifier.load_state_dict(torch.load(save_path))
purifier.eval()

# 5. Evaluate End-to-End Latency & Recovery
print("\n--- Evaluating Purifier Recovery ---")
from src.attacks import load_attacked_tensors, make_dataloader_from_tensors

fgsm_test_imgs, fgsm_test_labels = load_attacked_tensors(os.path.join(PROJECT_ROOT, 'results', f'fgsm_test_eps{TRAINING_EPS}.pt'))
pgd_test_imgs, pgd_test_labels = load_attacked_tensors(os.path.join(PROJECT_ROOT, 'results', f'pgd_test_eps{TRAINING_EPS}.pt'))

fgsm_loader = make_dataloader_from_tensors(fgsm_test_imgs, fgsm_test_labels, BATCH_SIZE)
pgd_loader = make_dataloader_from_tensors(pgd_test_imgs, pgd_test_labels, BATCH_SIZE)

# Wrapped Diagnostic loader for Chained Evaluation
class PurifiedDiagnosticLoader:
    def __init__(self, adv_loader, purifier, device):
        self.adv_loader = adv_loader
        self.purifier = purifier
        self.device = device
    def __iter__(self):
        for imgs, lbls in self.adv_loader:
            imgs = imgs.to(self.device)
            with torch.no_grad():
                purified = self.purifier(imgs)
            yield purified, lbls
    def __len__(self):
        return len(self.adv_loader)

# Latency Check
start_time = time.time()
with torch.no_grad():
    _ = purifier(fgsm_test_imgs[:1].to(device))
latency_ms = (time.time() - start_time) * 1000
print(f"Purifier Inference Latency: {latency_ms:.1f} ms (Target ~100ms)")

# Evaluate Chained Accuracy
metrics_fgsm = evaluate_model(diag_model, PurifiedDiagnosticLoader(fgsm_loader, purifier, device), device)
print_report(metrics_fgsm, title='Purifier -> Diagnostic (FGSM)')

metrics_pgd = evaluate_model(diag_model, PurifiedDiagnosticLoader(pgd_loader, purifier, device), device)
print_report(metrics_pgd, title='Purifier -> Diagnostic (PGD)')
