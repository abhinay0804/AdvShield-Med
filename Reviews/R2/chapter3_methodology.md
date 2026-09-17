# Chapter 3: Methodology and System Design

## 3.1 Proposed Methodology Overview

The multi-layered security envelope for a diagnostic neural network is called AdvShield-Med. The system blocks and cleans the incoming radiographs before they pass through a main diagnostic model - instead of retraining the diagnostic model to degrade the accuracy of clean images, as would be done with a retrained model.

The framework offers two modes of operation that can be selected by the operators:

```
MODE 1: STRICT REJECTION (Gatekeeper-First)
Input Radiograph (X) -> [ Preprocessing ] -> [ Gatekeeper CNN ] -> P(adv) < Threshold -> [ Diagnostic CNN ] -> Diagnosis
                                                                  -> P(adv) >= Threshold -> [ Security REJECT ] (Alert Logged)

MODE 2: SANITIZE-FIRST (Purifier-Assisted)
Input Radiograph (X) -> [ Purifier Autoencoder ] -> Cleaned Image -> [ Diagnostic CNN ] -> Diagnosis
                     -> (Parallel Logging) -> [ Gatekeeper CNN ] -> (Security Audit Log)
```

1. **Mode 1: Strict Rejection (Gatekeeper-First):** X-rays coming in pass through a Gatekeeper CNN as it comes. The system is trained to reject an image if the expected noise value for the image is above a certain limit imposed by an adversarial noise. If clean, then it scans the primary diagnostic network.
2. **Mode 2: Sanitize-First (Purifier-Assisted):** The X-rays sent to the U-Net Purifier are sanitized first and then classified. The Purifier removes high-frequency perturbation noise, while preserving lung structures. The image is then transferred to the diagnostic model when it is cleaned. Concurrently, the Gatekeeper registers security audit logs to the raw image, without interfering with diagnostic workflow.

## 3.2 System Architecture and Component Workflow

Combines attack generators, defenders, data loaders, an API server and a web interface into a single pipeline.

Pipeline Layers:

- **Clinical User Interface:** React 19 + TypeScript + Tailwind CSS 4 UI dashboard. Allows uploading of X-ray images via drag and drop, and real time security alerts (Clean vs Adversarial), confidence %, and diagnosis.
- **FastAPI Backend Server:** Async REST API Server (Python). Supports OAuth2 JWT authentication, decoding images to PyTorch tensors, pipeline execution context and SQL persistence.
- **AdvShield-Med Defense Engine:** (1) ResNet-18 Gatekeeper binary detector, (2) U-Net Purifier autoencoder, and (3) ResNet-18 Diagnostic baseline classifier.
- **PostgreSQL Database Container:** Docker container that contains the PostgreSQL 15 database. Maintains user credentials, stores and/or logs timestamp of scans, and controls access and audit logs.

## 3.3 Module Specifications

The proposed AdvShield-Med framework is split into eight modules, each written in software, in both data processing and security defence and deployment layers:

### 3.3.1 Data Management Module (data.py)
This module (data.py) is responsible for data management. Classifies the Kermany Pediatric Chest X Ray dataset [Kermany et al., 2018] (5,856 images, Normal y=0 vs Pneumonia y=1). The module re-splits the dataset into a train set, val set, and test set of 70%, 15% and 15% respectively. Grayscale X-rays are converted to 3-channel RGB, resized to 224x224, and normalized using ImageNet mean ($\mu = [0.485, 0.456, 0.406]$) and standard deviation ($\sigma = [0.229, 0.224, 0.225]$). The mini-batches are created with PyTorch WeightedRandomSampler for class balancing.

### 3.3.2 Baseline Diagnostic Module (models.py)
Fine-tuned ResNet-18 with pre-trained weights for binary classification of disease. A linear head is replaced by a linear(512, 2) head. For stage 1, the classification head is trained for 10 epochs with $\eta = 1	imes10^{-3}$. The following optimization phase is the unfreeze phase with $\eta = 1	imes10^{-5}$ for 30 epochs on all the layers. When fine-tuned, the weight of the model is set to `requires_grad = False` so as to keep the clean accuracy.

### 3.3.3 Adversarial Attack Module (attacks.py)
Generates white-box perturbations using wraps to torchattacks of the diagnostic model:
1. **Fast Gradient Sign Method (FGSM):** Single-step attack generating $x_{adv} = x + \epsilon \cdot 	ext{sign}(
abla_x \mathcal{L}(	heta, x, y))$ for $\epsilon \in \{0.01, 0.03, 0.06\}$.
2. **Projected Gradient Descent (PGD):** Multi-step attack (40 steps, step size $lpha = \epsilon / 10$) using PGD for same values of $\epsilon$.
3. **DeepFool:** Minimal perturbation boundary attack (50 steps, overshoot 0.02).

### 3.3.4 Gatekeeper Detection Module (models.py, train.py)
Adds a new module (models.py, train.py) to detect gating errors. ResNet-18 binary classifier (Linear(512, 2)) to classify clean images ($y=0$) from adversarial images ($y=1$). Mini-batches are split in half equally and half with FGSM perturbations ($\epsilon = 0.03$). Trained only on FGSM to test cross-attack generalization performance on the new PGD and DeepFool attacks.

### 3.3.5 Image Purification Module (purifier.py)
U-Net Convolutional Autoencoder with 4 encoder blocks (3 -> 32 -> 64 -> 128 -> 256), bottleneck (256 -> 512), and 4 decoder blocks (512 -> 256 -> 128 -> 64 -> 32) using skip connections. Trained using composite loss:
$$\mathcal{L}_{purifier} = 1.0 \cdot \mathcal{L}_{MSE} + 0.5 \cdot (1 - 	ext{SSIM}) + 0.1 \cdot \mathcal{L}_{perceptual}$$
MSE: Reconstruction of the image in the pixel level, SSIM: Preservation of lung morphology, VGG-16 perceptual loss: Preservation of the similarity of features.

### 3.3.6 Robust Backup Diagnostic Module (phase5b_backup.py)
Secondary ResNet-18 classifier trained using adversarial training using a 50/50 mixture of clean and FGSM-perturbed radiographs with disease labels ($y \in \{	ext{Normal}, 	ext{Pneumonia}\}$). Offers a backup diagnostic route for times when it is not possible to tolerate the latency of purification.

### 3.3.7 Pipeline Orchestration Module (pipeline.py)
Adds Gatekeeper, Purifier and Diagnostic models to AdvShieldPipeline. Checks that the diagnostic parameters have `requires_grad` set to `False` at initialization time. Uses `sweep_threshold()` to find the best threshold for decisions, $	au \in [0.3, 0.7]$, on the validation set.

### 3.3.8 Interpretability Module (phase5d_interpretability.py)
Visualizes the attention of the diagnostic backbone's final conv layer (`layer4[-1].conv2`) using Grad-CAM. Produces heat maps for all three types of images (clean, attacked and purified) to visualize spatial attention recovery.

## 3.4 Hardware, Software and Environment Stack

The architecture is multi-tiered and based on the following technical concepts:
1. **Deep Learning Engine:** PyTorch v2.2.0 and Torchvision v0.17.0 with tensor autograd and ResNet-18/VGG-16 backbones.
2. **Adversarial Attacks:** Torchattacks v3.5.1 (FGSM, PGD and DeepFool).
3. **Metrics:** Scikit-Learn v1.4.0 for accuracy, precision, recall, F1 and confusion matrices.
4. **Experiment Tracking:** Weights & Biases (WandB v0.16) and CSV loss curve loggers.
5. **REST API Server:** FastAPI v0.109.0 - for handling requests asynchronously, authentication with JWT, and context for model inference.
6. **Database Storage:** Store user accounts, scan audit history, and security logs using PostgreSQL 15 in Docker, through SQLAlchemy/Asyncpg.
7. **Web UI:** Drag and drop file upload, security notifications and diagnostics displays using React 19, TypeScript and Vite 8, and Tailwind CSS v4.0.
8. **Compute Hardware:** NVIDIA T4 GPUs (16 GB VRAM) in Google Colab for fine-tuning, attack generation and evaluation.

## 3.5 Feasibility, Risk Management, and Ethics

The identified and resolved four main technical risks were:
1. **GPU Memory/Quota Limits:** The timeouts of API calls are due to the 40-step PGD generation that occurs due to the GPU Memory/Quota limits. Mitigation: Store and pre-generate perturbed sets of tensors (.pt files).
2. **Cross-Attack Generalization Gap:** Gatekeeper fails to see PGD or DeepFool attacks that were out-of-scope from training. When using Sanitize-First mode, use U-Net Purifier autoencoder as an alternative in case it fails.
3. **Spatial Blurring from Autoencoders:** Lung opacities are blurred by using standard MSE loss. The mitigation method is to fuse the loss function of MSE and SSIM structural loss and perceptual feature loss of VGG-16.
4. **Class Imbalance:** Normally distributed Pneumonia data samples are smaller than Normal data samples, making predictions biased. Please put in place PyTorch WeightedRandomSampler and class-weighted CrossEntropyLoss.

## 3.6 Experimental Setup and Execution Phases

The validation of the empirical research goes through 6 phases that are structured:
- **Phase 1:** Optimize the baseline ResNet-18 model on the Kermany dataset, then test the model accuracy in the clean state (Target: Accuracies >= 85%).
- **Phase 2:** Create FGSM, PGD and DeepFooling attacks; measure loss of diagnostic accuracy.
- **Phase 3:** Train Gatekeeper only on an FGSM data set and test it on a clean, FGSM, and unseen PGD data set to measure the detection F1.
- **Phase 4:** Combine Gatekeeper model with Diagnostic model; sweep decision thresholds $	au \in [0.3, 0.7]$.
- **Phase 5:** Train Autoencoder U-Net for de-noise; Train Back-up model; Evaluate using DeepFool Attacks; Generate Grad-CAM heatmaps.
- **Phase 6:** Evaluate pipeline in Sanitize-First mode on test sets; show end-to-end Web dashboard.
