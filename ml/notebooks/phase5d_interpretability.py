import os, sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT = os.path.abspath('.')
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'chest_xray')
sys.path.insert(0, PROJECT_ROOT)

from src.data     import get_dataloaders
from src.models   import build_diagnostic_model, load_model
from src.purifier import PurifierCNN
import torchattacks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Models (we need gradients for the diagnostic model, so don't freeze it completely, just set to eval)
diag_model = load_model(build_diagnostic_model, os.path.join(PROJECT_ROOT, 'checkpoints', 'diagnostic_resnet18.pth'), device)
diag_model.eval()

purifier = PurifierCNN().to(device)
purifier.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'checkpoints', 'purifier.pth')))
purifier.eval()

# 2. Simple Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Backward pass for the target class
        target = output[0, class_idx]
        target.backward(retain_graph=True)
        
        # Global average pooling on gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations
        activations = self.activations[0] # (C, H, W)
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average across channels and ReLU
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)
        
        # Normalize between 0 and 1
        heatmap /= torch.max(heatmap)
        
        # Resize to original image size
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        
        return heatmap, class_idx

# Target the final convolutional layer in ResNet-18
target_layer = diag_model.layer4[-1].conv2
cam = GradCAM(diag_model, target_layer)

# 3. Get a sample True Positive (Pneumonia) image
loaders, _ = get_dataloaders(DATA_ROOT, batch_size=1, num_workers=0)
clean_img = None
for img, lbl in loaders['test']:
    if lbl.item() == 1: # Pneumonia
        clean_img = img.to(device)
        break

# 4. Generate the 3 scenarios
# Scenario 1: Clean Image
heatmap_clean, pred_clean = cam(clean_img, class_idx=1)

# Scenario 2: Attacked Image (FGSM)
attack = torchattacks.FGSM(diag_model, eps=0.03)
attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
with torch.enable_grad():
    adv_img = attack(clean_img, torch.tensor([1]).to(device))
heatmap_adv, pred_adv = cam(adv_img, class_idx=1)

# Scenario 3: Purified Image
with torch.no_grad(): # Purifier doesn't need gradients for inference
    purified_img = purifier(adv_img)
# Pass purified image through CAM (requires grad for diagnostic model)
purified_img.requires_grad_(True)
heatmap_purified, pred_purified = cam(purified_img, class_idx=1)

# 5. Visualization Helper
def overlay_heatmap(img_tensor, heatmap):
    # Un-normalize image for display
    img = img_tensor.cpu().squeeze().detach().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap)[..., :3] # Extract RGB channels
    
    overlay = heatmap_colored * 0.4 + img
    overlay = np.clip(overlay, 0, 1) # Ensure valid range
    return img, overlay

# 6. Plotting
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

scenarios = [
    ("Clean Image", clean_img, heatmap_clean, pred_clean),
    ("Attacked Image (FGSM)", adv_img, heatmap_adv, pred_adv),
    ("Purified Image", purified_img, heatmap_purified, pred_purified)
]

for i, (title, img_t, hm, pred) in enumerate(scenarios):
    img_disp, overlay = overlay_heatmap(img_t, hm)
    
    axes[i, 0].imshow(img_disp)
    pred_str = "Pneumonia" if pred == 1 else "Normal"
    axes[i, 0].set_title(f"{title}\nPrediction: {pred_str}")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(overlay)
    axes[i, 1].set_title(f"Grad-CAM Heatmap")
    axes[i, 1].axis('off')

plt.tight_layout()
plots_dir = os.path.join(PROJECT_ROOT, 'training_plots')
os.makedirs(plots_dir, exist_ok=True)
save_path = os.path.join(plots_dir, 'phase5d_gradcam.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"Grad-CAM visualization saved successfully -> {save_path}")
