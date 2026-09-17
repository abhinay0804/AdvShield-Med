import json
import os

notebook_path = '/mnt/shared/Projects/PJT/advshield-med/notebooks/phase3_gatekeeper.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code': continue
    src = "".join(cell['source'])
    
    # 2. Dataset Generation Cell
    if "class GatekeeperDataLoaderWrapper:" in src:
        src = """class GatekeeperDataLoaderWrapper:
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
"""

    # 3. Graph Saving Cell
    if "plt.savefig" in src:
        src = src.replace("plt.show()", "")
        src = src.replace("os.path.join(RESULTS_DIR, 'phase3_curves.png')", "os.path.join('training_plots', 'phase3_curves.png')")
        src = "import os\nos.makedirs('training_plots', exist_ok=True)\n" + src

    # 4. Skip Training Loop
    if "load_state_dict" in src and "Skipping" in src:
        src = "print('Skipping training loop and loading best checkpoint...')\ngatekeeper = build_gatekeeper_model()\ngatekeeper = gatekeeper.to(device)\ngatekeeper.load_state_dict(torch.load(GATE_CHECKPOINT))\n"

    cell['source'] = [line + '\n' for line in src.split('\n')]
    if cell['source'] and cell['source'][-1].endswith('\n\n'):
        cell['source'][-1] = cell['source'][-1][:-1]
        
    cell['source'] = [s for s in cell['source'] if s]

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
