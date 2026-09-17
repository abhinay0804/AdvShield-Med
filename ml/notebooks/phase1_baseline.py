# ─── Colab: mount Drive and set paths ────────────────────────────────────────
import os, sys

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = '/content/drive/MyDrive/advshield-med'
    DATA_ROOT    = '/content/drive/MyDrive/chest_xray'
    sys.path.insert(0, PROJECT_ROOT)
else:
    try:
        # When running as a python script
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        # When running interactively in Jupyter from the project root
        PROJECT_ROOT = os.path.abspath('.')
    DATA_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'chest_xray')
    sys.path.insert(0, PROJECT_ROOT)

os.makedirs(os.path.join(PROJECT_ROOT, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)

print(f'PROJECT_ROOT : {PROJECT_ROOT}')
print(f'DATA_ROOT    : {DATA_ROOT}')

# ─── Install dependencies (Colab only) ───────────────────────────────────────
if IN_COLAB:
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'torchattacks', 'wandb', 'scikit-learn', 'tqdm'])
    print('Dependencies installed.')

import torch
from src.data   import get_dataloaders
from src.models import build_diagnostic_model, freeze_model, save_model, count_parameters, freeze_backbone, unfreeze_backbone
from src.train  import train
from src.evaluate import evaluate_model, print_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'diagnostic_resnet18.pth')
LOG_CSV    = os.path.join(PROJECT_ROOT, 'results', 'phase1_log.csv')
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 prevents multiprocessing crash when running as a script

loaders, class_weights = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
# Prints split summary automatically

model = build_diagnostic_model(pretrained=True)
print(f'Trainable parameters: {count_parameters(model):,}')

# ── Stage 1: Train Head Only (High LR) ───────────────────────────────────
print('\n--- Stage 1: Training Classification Head ---')
model = freeze_backbone(model)
model = train(
    model=model,
    loaders=loaders,
    checkpoint_path=CHECKPOINT.replace('.pth', '_stage1.pth'),
    log_csv_path=LOG_CSV.replace('.csv', '_stage1.csv'),
    run_name='phase1_diagnostic_stage1',
    num_epochs=10,
    learning_rate=1e-3,
    class_weights=class_weights,
    use_amp=True,
    use_cosine_scheduler=True,
    early_stopping_patience=3,
    device=device,
    use_wandb=True,
    wandb_project='advshield-med',
)

# ── Stage 2: Full Fine-Tuning (Low LR) ───────────────────────────────────
print('\n--- Stage 2: Full Network Fine-Tuning ---')
model = unfreeze_backbone(model)
model = train(
    model=model,
    loaders=loaders,
    checkpoint_path=CHECKPOINT,
    log_csv_path=LOG_CSV,
    run_name='phase1_diagnostic_stage2',
    num_epochs=30,
    learning_rate=1e-5,
    class_weights=class_weights,
    use_amp=True,
    use_cosine_scheduler=True,
    early_stopping_patience=5,
    device=device,
    use_wandb=True,
    wandb_project='advshield-med',
)

import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv(LOG_CSV)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(log['epoch'], log['train_loss'], label='Train')
axes[0].plot(log['epoch'], log['val_loss'],   label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves — Diagnostic Model'); axes[0].legend()

axes[1].plot(log['epoch'], log['train_acc'], label='Train')
axes[1].plot(log['epoch'], log['val_acc'],   label='Val')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curves — Diagnostic Model'); axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'phase1_curves.png'), dpi=150)
plt.show()

metrics = evaluate_model(model, loaders['test'], device=device)
print_report(metrics, title='Diagnostic Model — Clean Test Set')

# ── Success check ────────────────────────────────────────────────────────────
if metrics['accuracy'] >= 0.85:
    print('\n✓ SUCCESS: Accuracy ≥ 85% — Phase 1 target met.')
else:
    print(f'\n⚠ WARNING: Accuracy {metrics["accuracy"]:.4f} < 85% target.')
    print('  Consider: longer training, LR adjustment, or check data loading.')

model = freeze_model(model)
print(f'Trainable params after freeze: {count_parameters(model)}')
assert count_parameters(model) == 0, 'Model not fully frozen!'
print('\n✓ Diagnostic model is frozen. Phase 1 complete.')
print(f'  Checkpoint: {CHECKPOINT}')
