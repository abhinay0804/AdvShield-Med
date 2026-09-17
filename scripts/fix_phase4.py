import json
import os

notebook_path = '/mnt/shared/Projects/PJT/advshield-med/notebooks/phase4_pipeline.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code': continue
    src = "".join(cell['source'])
    
    # 1. Setup Cell
    if "loaders = get_dataloaders(" in src or "IN_COLAB" in src:
        src = src.replace("os.path.dirname(os.path.dirname(os.path.abspath('__file__')))", "os.path.abspath('.')")
        src = src.replace("batch_size=BATCH_SIZE)", "batch_size=BATCH_SIZE, num_workers=0)")
        src = src.replace("loaders = get_dataloaders", "loaders, _ = get_dataloaders")
        
    # 2. Add validation generation right before loading
    if "fgsm_val_imgs, fgsm_val_labels = load_attacked_tensors" in src:
        generation_code = """
fgsm_val_path = os.path.join(RESULTS_DIR, f'fgsm_val_eps{TRAINING_EPS}.pt')
if not os.path.exists(fgsm_val_path):
    print("Generating missing FGSM validation set...")
    from src.attacks import generate_attacked_dataset, save_attacked_tensors
    imgs, lbls = generate_attacked_dataset(diag_model, val_loader, 'fgsm', eps=TRAINING_EPS, device=device)
    save_attacked_tensors(imgs, lbls, fgsm_val_path)

"""
        src = generation_code + src
        
    # 3. Graph Saving
    if "plt.savefig" in src:
        src = src.replace("plt.show()", "")
        src = src.replace("os.path.join(RESULTS_DIR, 'phase4_threshold_curve.png')", "os.path.join('training_plots', 'phase4_threshold_curve.png')")

    cell['source'] = [line + '\n' for line in src.split('\n')]
    if cell['source'] and cell['source'][-1].endswith('\n\n'):
        cell['source'][-1] = cell['source'][-1][:-1]
        
    cell['source'] = [s for s in cell['source'] if s]

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
