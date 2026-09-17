import os, sys, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

PROJECT_ROOT = os.path.abspath('.')
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'chest_xray')
sys.path.insert(0, PROJECT_ROOT)

from src.data     import get_dataloaders
from src.models   import build_diagnostic_model, load_model, freeze_model
from src.evaluate import evaluate_model, print_report
from src.attacks  import generate_attacked_dataset, save_attacked_tensors, load_attacked_tensors, make_dataloader_from_tensors
from src.purifier import PurifierCNN
from notebooks.phase5a_purifier import PurifiedDiagnosticLoader # Reuse our wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 32

# 1. Models Initialization
print("Loading all models for evaluation...")
diag_model = load_model(build_diagnostic_model, os.path.join(PROJECT_ROOT, 'checkpoints', 'diagnostic_resnet18.pth'), device)
diag_model = freeze_model(diag_model)

gatekeeper = load_model(build_diagnostic_model, os.path.join(PROJECT_ROOT, 'checkpoints', 'gatekeeper_resnet18.pth'), device)
gatekeeper = freeze_model(gatekeeper)

backup_model = load_model(build_diagnostic_model, os.path.join(PROJECT_ROOT, 'checkpoints', 'backup_resnet18.pth'), device)
backup_model = freeze_model(backup_model)

purifier = PurifierCNN().to(device)
purifier.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'checkpoints', 'purifier.pth')))
purifier.eval()

# 2. Generate Subset Dataset
loaders, _ = get_dataloaders(DATA_ROOT, batch_size=1, num_workers=0) # Batch 1 for easy filtering
test_loader = loaders['test']

# Target: 54 Normal (0), 146 Pneumonia (1) to perfectly stratify the 27%/73% ratio of the full test set
target_0 = 54
target_1 = 146
collected_0 = 0
collected_1 = 0

subset_images = []
subset_labels = []

print("\nCollecting 200 stratified test images...")
for img, lbl in test_loader:
    label_val = lbl.item()
    if label_val == 0 and collected_0 < target_0:
        subset_images.append(img)
        subset_labels.append(lbl)
        collected_0 += 1
    elif label_val == 1 and collected_1 < target_1:
        subset_images.append(img)
        subset_labels.append(lbl)
        collected_1 += 1
        
    if collected_0 == target_0 and collected_1 == target_1:
        break

subset_images = torch.cat(subset_images, dim=0)
subset_labels = torch.cat(subset_labels, dim=0)
print(f"Collected {len(subset_images)} images (Normal: {collected_0}, Pneumonia: {collected_1})")

subset_dataset = TensorDataset(subset_images, subset_labels)
subset_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Generate DeepFool Attacks
deepfool_path = os.path.join(PROJECT_ROOT, 'results', 'deepfool_test_subset.pt')

if not os.path.exists(deepfool_path):
    print("\nGenerating DeepFool attacks... (This may take a few minutes)")
    start_time = time.time()
    adv_images, adv_labels = generate_attacked_dataset(
        model=diag_model,
        loader=subset_loader,
        attack_type="deepfool",
        eps=0.0, # Ignored by DeepFool
        device=device
    )
    save_attacked_tensors(adv_images, adv_labels, deepfool_path)
    print(f"DeepFool generation took {(time.time() - start_time):.1f} seconds")
else:
    adv_images, adv_labels = load_attacked_tensors(deepfool_path)

adv_loader = make_dataloader_from_tensors(adv_images, adv_labels, BATCH_SIZE)

# 4. Evaluation Suite
print("\n=== Evaluating DeepFool (Unseen Attack Phase 5) ===")

# Undefended Model
undefended_metrics = evaluate_model(diag_model, adv_loader, device)
print_report(undefended_metrics, title="Undefended Diagnostic Model (DeepFool)")

# Gatekeeper (Detecting adversarial nature, true labels map to 1 for adversarial)
# We map all adv_labels to 1 to test detection
gate_labels = torch.ones_like(adv_labels)
gate_loader = make_dataloader_from_tensors(adv_images, gate_labels, BATCH_SIZE)
gate_metrics = evaluate_model(gatekeeper, gate_loader, device)
print_report(gate_metrics, title="Gatekeeper Detection (DeepFool)")

# Purifier Recovery
purified_loader = PurifiedDiagnosticLoader(adv_loader, purifier, device)
purifier_metrics = evaluate_model(diag_model, purified_loader, device)
print_report(purifier_metrics, title="Purifier Recovery Path (DeepFool)")

# Backup Model Recovery
backup_metrics = evaluate_model(backup_model, adv_loader, device)
print_report(backup_metrics, title="Backup Model Recovery Path (DeepFool)")

# 5. Summary Table
print("\n=== Phase 5 DeepFool Summary ===")
df = pd.DataFrame({
    'Metric': [
        'Undefended Accuracy',
        'Gatekeeper Detection (Recall)',
        'Purifier Recovered Accuracy',
        'Backup Model Accuracy'
    ],
    'Value (DeepFool)': [
        f"{undefended_metrics['accuracy']:.2%}",
        f"{gate_metrics['recall']:.2%}", # Recall because class 1 is "adversarial"
        f"{purifier_metrics['accuracy']:.2%}",
        f"{backup_metrics['accuracy']:.2%}"
    ]
})
print(df.to_string(index=False))
