# AdvShield-Med: Multi-Attack Adversarial Detection & Defense Framework for Chest X-Ray Diagnosis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)]()

> **AdvShield-Med** is a pre-inference security wrapper and image purification framework designed to protect deep learning CAD (Computer-Aided Diagnosis) systems against adversarial attacks. It inspects and sanitizes incoming radiological X-rays *before* they reach an unmodified diagnostic model—achieving high cross-attack generalization even against unseen iterative attacks.

---

## 📌 Key Benchmarks & Quantitative Results

| Evaluation Environment | Diagnostic Accuracy | Attack Catch Rate (F1) | Diagnostic Recovery Rate | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Chest X-Rays (Baseline)** | **96.13%** | — | — | Baseline High Clinical Accuracy |
| **Under FGSM Attack ($\epsilon=0.03$)** | **71.33%** | **1.000 (100%)** | **88.6%** | High Detection & Recovery |
| **Under PGD-40 Attack ($\epsilon=0.03$)** | **0.00%** | **0.996 (99.2%)** | **84.2%** | **Full Cross-Attack Generalization** |

* **Zero-Shot Cross-Attack Generalization:** The Gatekeeper model was trained **ONLY on single-step FGSM noise**, yet successfully detected **99.2% of unseen 40-step PGD attacks** ($F_1 = 0.996$) with a **100% clean image pass-through rate**.
* **Diagnostic Recovery:** The U-Net Purifier restored diagnostic accuracy from **0.00% back to 84.2%** on heavily perturbed PGD X-rays.

---

## 🏗️ System Architecture & Workflow

```
                  ┌─────────────────────────────────────┐
                  │   Input Chest Radiograph (224x224)  │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │  Gatekeeper CNN (ResNet-18 Model)   │ ◄── Evaluates Perturbation Level
                  └──────────────────┬──────────────────┘
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
      [Decision: CLEAN]                           [Decision: ADVERSARIAL]
               │                                           │
               │                            ┌──────────────┴──────────────┐
               │                            ▼                             ▼
               │                   [Strict Rejection]             [Sanitize-First]
               │                   Audit Log Generated            U-Net Purifier Engine
               │                            │                             │
               │                            │                             ▼
               │                            │                 ┌──────────────────────┐
               │                            │                 │  U-Net Autoencoder   │
               │                            │                 │  (Noise Stripped)    │
               │                            │                 └──────────┬───────────┘
               │                            │                            │
               ▼                            │                            ▼
  ┌──────────────────────────┐              │              ┌──────────────────────────┐
  │  Diagnostic ResNet-18    │              │              │  Diagnostic ResNet-18    │
  │    (Frozen Weights)      │              │              │    (Frozen Weights)      │
  └────────────┬─────────────┘              │              └────────────┬─────────────┘
               │                            │                           │
               ▼                            ▼                           ▼
  ┌──────────────────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
  │ Clean Pneumonia Diagnosis│    │ Input Rejected   │    │ Purified Diagnosis       │
  └──────────────────────────┘    └──────────────────┘    └──────────────────────────┘
```

---

## 🧪 Technical Innovations & Custom Engineering

### 1. 2-Stage Transfer Learning Protocol
To prevent catastrophic forgetting of ImageNet features when adapting to radiological images:
* **Stage 1 (Head Warmup):** Convolutional backbone is frozen (`requires_grad = False`); only the classification head (`Linear(512, 2)`) is trained for 10 epochs at $\eta = 10^{-3}$.
* **Stage 2 (Full Alignment):** Entire network is unfrozen and fine-tuned for 30 epochs using a low learning rate ($\eta = 10^{-5}$) with a **Cosine Annealing Scheduler** decaying to $10^{-7}$.
* **Safety Freezing:** Post-training, diagnostic parameters are permanently frozen with runtime safety assertions.

### 2. Custom 3-Term Composite Loss Function ($\mathcal{L}_{\text{purifier}}$)
Standard MSE loss causes severe blurring in medical autoencoders. We formulated a tri-level composite loss:
$$\mathcal{L}_{\text{purifier}} = 1.0 \cdot \mathcal{L}_{\text{MSE}} + 0.5 \cdot (1 - \text{SSIM}) + 0.1 \cdot \mathcal{L}_{\text{Perceptual}}$$
* **Pixel Loss ($\mathcal{L}_{\text{MSE}}$):** Maintains overall luminance and intensity alignment.
* **Structural Loss ($\mathcal{L}_{\text{SSIM}}$):** Preserves sharp bone boundaries and lung field contours ($11 \times 11$ Gaussian window).
* **Perceptual Feature Loss ($\mathcal{L}_{\text{Perceptual}}$):** Computes deep feature map distance using a frozen VGG-16 backbone at layer `block2_conv2`, guaranteeing high-level diagnostic feature retention.

---

## 📁 Repository Structure

```
advshield-med/
├── backend/                  # FastAPI REST API server
│   ├── app/
│   │   ├── api/             # Diagnostic & purification endpoints
│   │   └── core/            # Pipeline loader & inference engine
│   └── main.py              # Server entry point
├── frontend/                 # React 19 + Vite + Tailwind CSS Dashboard
│   ├── src/
│   │   ├── components/      # Radiologist upload, heatmap & metrics components
│   │   └── App.jsx
├── ml/                       # Core Machine Learning Framework
│   ├── src/
│   │   ├── models.py        # ResNet-18 builders & freeze utilities
│   │   ├── purifier.py      # U-Net Autoencoder & Combined Loss
│   │   ├── pipeline.py      # AdvShieldPipeline orchestration engine
│   │   ├── attacks.py       # FGSM, PGD-40 & DeepFool attack harness
│   │   ├── train.py         # Multi-stage training loop with Cosine Annealing
│   │   ├── data.py          # Stratified dataset loaders
│   │   └── evaluate.py     # Evaluation & metrics tools
│   ├── notebooks/           # Phase execution scripts & notebooks
│   │   ├── phase1_baseline.py
│   │   ├── phase2_attacks.py
│   │   ├── phase3_gatekeeper.py
│   │   ├── phase4_pipeline.py
│   │   ├── phase5a_purifier.py
│   │   └── phase6_final_pipeline_eval.py
│   ├── checkpoints/         # Saved PyTorch model weights (.pth)
│   └── results/             # Benchmark metrics, CSV logs, & training plots
├── scripts/                  # Helper & maintenance scripts
├── docker-compose.yml        # Multi-container deployment configuration
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project Documentation
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites & Installation

```bash
# Clone repository
git clone https://github.com/NamaAbhinay/advshield-med.git
cd advshield-med

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the [Kermany Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and structure it as follows:
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/
```

### 3. Run Pipeline Evaluation
```bash
# Run pipeline evaluation script
python ml/notebooks/phase4_pipeline.py
```

### 4. Run Full-Stack Web Application

```bash
# Start FastAPI backend
cd backend
uvicorn main:app --reload --port 8000

# In a new terminal, start React frontend
cd frontend
npm install
npm run dev
```

---

## 👥 Authors & Acknowledgments

* **Nama Abhinay** (23BPS1080)
* **Ryali Lakshman** (23BPS1026)
* **Guide:** Dr. Kalaipriyan Thiru
* **Institution:** Vellore Institute of Technology (VIT)

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
