import os, sys
import torch
import pandas as pd
import matplotlib.pyplot as plt

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

from src.data     import get_dataloaders
from src.models   import (build_diagnostic_model, build_gatekeeper_model,
                           freeze_model, load_model)
from src.attacks  import load_attacked_tensors, make_dataloader_from_tensors
from src.evaluate import evaluate_model, print_report, build_results_row
from src.pipeline import AdvShieldPipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIAG_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'diagnostic_resnet18.pth')
GATE_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'gatekeeper_resnet18.pth')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'results')
TRAINING_EPS    = 0.03
BATCH_SIZE      = 32

print(f'Device: {device}')


diag_model = load_model(build_diagnostic_model, DIAG_CHECKPOINT, device=device)
diag_model = freeze_model(diag_model)

gatekeeper = load_model(build_gatekeeper_model, GATE_CHECKPOINT, device=device)
# Gatekeeper is eval-only here — no need to freeze (not training anything in Phase 4)

print('Both models loaded.')


loaders, _ = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=0)
val_loader  = loaders['val']
test_loader = loaders['test']

# Load pre-generated attack tensors from Phase 2
fgsm_test_imgs, fgsm_test_labels = load_attacked_tensors(
    os.path.join(RESULTS_DIR, f'fgsm_test_eps{TRAINING_EPS}.pt')
)
pgd_test_imgs, pgd_test_labels = load_attacked_tensors(
    os.path.join(RESULTS_DIR, f'pgd_test_eps{TRAINING_EPS}.pt')
)
fgsm_test_loader = make_dataloader_from_tensors(fgsm_test_imgs, fgsm_test_labels, BATCH_SIZE)
pgd_test_loader  = make_dataloader_from_tensors(pgd_test_imgs,  pgd_test_labels,  BATCH_SIZE)



fgsm_val_path = os.path.join(RESULTS_DIR, f'fgsm_val_eps{TRAINING_EPS}.pt')
if not os.path.exists(fgsm_val_path):
    print("Generating missing FGSM validation set...")
    from src.attacks import generate_attacked_dataset, save_attacked_tensors
    imgs, lbls = generate_attacked_dataset(diag_model, val_loader, 'fgsm', eps=TRAINING_EPS, device=device)
    save_attacked_tensors(imgs, lbls, fgsm_val_path)

# Load val attack tensors (generated in Phase 3)
fgsm_val_imgs, fgsm_val_labels = load_attacked_tensors(
    os.path.join(RESULTS_DIR, f'fgsm_val_eps{TRAINING_EPS}.pt')
)
fgsm_val_loader = make_dataloader_from_tensors(fgsm_val_imgs, fgsm_val_labels, BATCH_SIZE)

# Temporary pipeline with default threshold (will be updated after sweep)
pipeline_tmp = AdvShieldPipeline(
    gatekeeper=gatekeeper,
    diagnostic=diag_model,
    threshold=0.5,
    device=device,
)

sweep_results = pipeline_tmp.sweep_threshold(
    clean_loader=val_loader,
    adversarial_loader=fgsm_val_loader,
    thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
)

# Save sweep table
sweep_df = pd.DataFrame(sweep_results)
sweep_df.to_csv(os.path.join(RESULTS_DIR, 'phase4_threshold_sweep.csv'), index=False)


# Plot clean pass rate vs adversarial detection rate tradeoff
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sweep_df['threshold'], sweep_df['clean_pass_rate'],  'b-o', label='Clean Pass Rate')
ax.plot(sweep_df['threshold'], sweep_df['adv_detect_rate'],  'r-o', label='Adv Detect Rate')
ax.set_xlabel('Threshold'); ax.set_ylabel('Rate')
ax.set_title('Threshold Sweep — Clean Pass Rate vs Adversarial Detection Rate')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('training_plots', 'phase4_threshold_curve.png'), dpi=150)



# ── SELECT YOUR THRESHOLD HERE ────────────────────────────────────────────────
# Look at the curve above and pick the threshold that gives the best tradeoff.
# Typical choice: high adv_detect_rate without dropping clean_pass_rate below ~0.90

CHOSEN_THRESHOLD = 0.5   # <── UPDATE THIS after inspecting the sweep table
print(f'Chosen threshold: {CHOSEN_THRESHOLD}')
print('This value must be reported and justified in your results section.')


pipeline = AdvShieldPipeline(
    gatekeeper=gatekeeper,
    diagnostic=diag_model,
    threshold=CHOSEN_THRESHOLD,
    device=device,
)
print(f'Pipeline ready with threshold={CHOSEN_THRESHOLD}')


print('\n=== Pipeline Evaluation on Test Set ===')

clean_results = pipeline.evaluate_batch(test_loader,      true_image_type='clean')
fgsm_results  = pipeline.evaluate_batch(fgsm_test_loader, true_image_type='adversarial')
pgd_results   = pipeline.evaluate_batch(pgd_test_loader,  true_image_type='adversarial')

for label, res in [('Clean', clean_results), ('FGSM (seen)', fgsm_results), ('PGD (unseen)', pgd_results)]:
    print(f'\n--- {label} ---')
    for k, v in res.items():
        if v is not None:
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')


# Pull Phase 1 and 2 metrics from saved CSVs
phase2_df = pd.read_csv(os.path.join(RESULTS_DIR, 'phase2_attack_results.csv'))
phase3_df = pd.read_csv(os.path.join(RESULTS_DIR, 'phase3_gatekeeper_results.csv'))

def get_metric(df, label_contains, metric):
    row = df[df['label'].str.contains(label_contains, case=False)]
    return row.iloc[0][metric] if len(row) > 0 else 'N/A'

final_table = [
    {'Metric': 'Diagnostic model accuracy — clean images',
     'Value': get_metric(phase2_df, 'Clean', 'accuracy')},
    {'Metric': f'Diagnostic model accuracy — under FGSM (ε={TRAINING_EPS})',
     'Value': get_metric(phase2_df, f'FGSM ε={TRAINING_EPS}', 'accuracy')},
    {'Metric': f'Diagnostic model accuracy — under PGD (ε={TRAINING_EPS})',
     'Value': get_metric(phase2_df, f'PGD ε={TRAINING_EPS}', 'accuracy')},
    {'Metric': 'Gatekeeper detection F1 — FGSM (seen attack)',
     'Value': get_metric(phase3_df, 'FGSM', 'f1')},
    {'Metric': 'Gatekeeper detection F1 — PGD (unseen, generalization test)',
     'Value': get_metric(phase3_df, 'PGD', 'f1')},
    {'Metric': 'End-to-end pipeline — clean images (gate accuracy)',
     'Value': f"{clean_results['gate_accuracy']:.4f}"},
    {'Metric': 'End-to-end pipeline — FGSM detection rate',
     'Value': f"{fgsm_results['rejection_rate']:.4f}"},
    {'Metric': 'End-to-end pipeline — PGD detection rate',
     'Value': f"{pgd_results['rejection_rate']:.4f}"},
    {'Metric': 'Pipeline rejection threshold (tuned)',
     'Value': str(CHOSEN_THRESHOLD)},
]

final_df = pd.DataFrame(final_table)
print(final_df.to_string(index=False))
final_df.to_csv(os.path.join(RESULTS_DIR, 'phase4_final_results.csv'), index=False)
print('\n✓ Final results table saved → results/phase4_final_results.csv')


from sklearn.metrics import f1_score, accuracy_score
from src.evaluate import evaluate_model

# Spot-check: compare our evaluate_model output vs direct sklearn on clean test set
our_metrics = evaluate_model(diag_model, test_loader, device=device)

sklearn_acc = accuracy_score(our_metrics['all_labels'], our_metrics['all_preds'])
sklearn_f1  = f1_score(our_metrics['all_labels'], our_metrics['all_preds'])

assert abs(our_metrics['accuracy'] - sklearn_acc) < 1e-6, 'Accuracy mismatch!'
assert abs(our_metrics['f1']       - sklearn_f1)  < 1e-6, 'F1 mismatch!'

print('✓ Metric sanity check passed — our evaluate_model matches sklearn directly.')

