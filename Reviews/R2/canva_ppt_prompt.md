# Canva AI Presentation Prompt & Slide-by-Slide Guide (Review 2)

**Course:** BCSE497J, Project-I (Fall 2026-2027)
**Review:** Review 2, Panel Review (20 Marks)
**Project Title:** AdvShield-Med: A Multi-Attack Adversarial Detection and Defense Framework for Robust Deep Learning-Based Chest X-Ray Diagnosis

---

## Part 1: Single Master Canva AI Prompt

> Copy and paste the text block below directly into **Canva Magic Design / AI Presentation Generator**:

```text
Create a 15-slide academic slide deck for a B.Tech Computer Science senior design project presentation titled "AdvShield-Med: A Multi-Attack Adversarial Detection and Defense Framework for Robust Deep Learning-Based Chest X-Ray Diagnosis".

Design aesthetic: Professional dark navy-blue background (#0A1128), crisp white body text (#F8F9FA), electric cyan accent colors (#00F5D4), subtle glassmorphism cards, modern clean typography (Inter or Montserrat font), minimal clutter, high visual impact. Use diagrams, flowcharts, structural tables, and icon callouts instead of long paragraphs.

Slide outline:
1. Title Slide: Project title, subtitle "Review 2: Domain, Literature Review, Methodology & System Design", student names, supervisor name, department of Computer Science and Engineering, VIT University.
2. Agenda: Bulleted roadmap covering Domain, Literature Review (15 recent papers 2020-2026), Problem Definition, Objectives & Scope, Proposed Methodology (Two-Part Framework), System Architecture & Modules, Feasibility & Risk Management, Project Timeline.
3. Domain Understanding: Growing role of Deep Learning CNNs in radiology (Chest X-Ray CAD systems), clinical benefits, and the emerging threat of adversarial cyber-attacks in healthcare.
4. Problem Motivation: Explain how invisible mathematical perturbations (FGSM, PGD, DeepFool) cause high-confidence misclassifications (Pneumonia vs Normal), posing direct risks to patient safety.
5. Problem Statement: Formal definition highlighting 3 gaps: lack of integrated defense frameworks, absence of cross-attack generalization testing, and clean accuracy degradation in adversarially trained models.
6. Objectives & Outcomes: 6 core project objectives (baseline diagnostic model, vulnerability benchmark, Gatekeeper detector, Purifier autoencoder, end-to-end pipeline, full-stack deployment).
7. Scope & Boundary Conditions: In-scope elements (Kermany dataset, ResNet-18, FGSM/PGD/DeepFool, Gatekeeper, Purifier, FastAPI, React) vs Out-of-scope elements (hospital PACS integration, federated learning, black-box transfer).
8. Literature Review Overview: Categorization of 15 recent papers (2020-2026) across 4 themes: Attack Vulnerabilities, Frequency/Intermediate Detection, Autoencoder Purification, and Multi-Layer Defenses.
9. Literature Comparison Table: Clean comparative table summarizing key literature (Author/Year, Focus, Dataset, Key Finding, Limitation).
10. Identified Research Gap: 4 critical gaps in existing literature leading to the motivation for AdvShield-Med.
11. Proposed Methodology, Two-Part Framing: Visual diagram showing Part A (Adversarial Detection via Gatekeeper CNN) and Part B (Adversarial Purification & Recovery via U-Net Autoencoder).
12. System Architecture: Full system block diagram showing Input Radiograph → Preprocessing → Gatekeeper Filter → Purifier Reconstruction → Diagnostic Model → FastAPI Backend → React UI.
13. Module Design & Technology Stack: Table of 8 software modules alongside tech stack icons (PyTorch, Torchattacks, FastAPI, React, PostgreSQL, Docker).
14. Feasibility, Ethics & Risk Register: Risk matrix table detailing 4 technical risks (GPU quota, cross-attack generalization, blur in purification, dataset imbalance) with concrete mitigation strategies. Ethical compliance note.
15. Conclusion & Q&A Slide: Summary of design readiness, milestones ahead, thank you message, open for panel questions.
```

---

## Part 2: Slide-by-Slide Content & Speaker Notes

### Slide 1: Title Slide
* **Title:** AdvShield-Med: A Multi-Attack Adversarial Detection and Defense Framework for Robust Deep Learning-Based Chest X-Ray Diagnosis
* **Subtitle:** Review 2, Domain, Literature Review, Methodology & System Design
* **Team Members:** [Student Name 1 - Reg. No.] | [Student Name 2 - Reg. No.]
* **Project Guide:** [Guide Name, Designation]
* **Department:** School of Computer Science and Engineering (SCOPE), VIT Chennai
* **Speaker Notes:** *"Good morning respected panel members. Today we present Review 2 for our Project-I titled AdvShield-Med, focusing on securing deep learning-based chest X-ray diagnostic systems against adversarial cyber-attacks."*

---

### Slide 2: Agenda
* **Key Sections:**
 1. Domain Context & Clinical Motivation
 2. Problem Statement & Research Gap
 3. Review of Recent Literature (15 Papers, 2020-2026)
 4. Project Objectives & Scope
 5. Proposed Methodology (Part A: Detection & Part B: Purification)
 6. System Architecture & Module Design
 7. Feasibility, Risk Management & Timeline
* **Speaker Notes:** *"This presentation covers the first 7 evaluation parameters for Review 2, detailing our domain analysis, literature review of 15 recent papers, proposed architecture, and implementation planning."*

---

### Slide 3: Domain Understanding, Medical AI & CAD Systems
* **Points:**
 * **Clinical Context:** Chest X-rays are the most frequently ordered diagnostic imaging modality worldwide.
 * **Deep Learning Integration:** Convolutional Neural Networks (ResNet, DenseNet) achieve near-radiologist performance in binary pneumonia classification.
 * **The Vulnerability:** Deep neural networks are non-linear mathematical functions susceptible to input perturbations.
 * **Clinical Significance:** In medical imaging, misclassification directly impacts patient safety, treatment decisions, and hospital workflow.
* **Speaker Notes:** *"Deep learning has revolutionized chest X-ray interpretation. However, because these networks rely on complex feature maps, minor pixel perturbations can manipulate their predictions, creating a severe vulnerability in clinical environments."*

---

### Slide 4: Problem Motivation, The Adversarial Threat
* **Visual:** Diagram showing `Clean X-Ray (Normal)` + `Invisible Noise ($\epsilon=0.03$)` $\rightarrow$ `Attacked Image` $\rightarrow$ `Model Prediction: Pneumonia (99% confidence)`.
* **Points:**
 * **Human-Imperceptible:** Noise magnitude $\epsilon$ is visually undetectable to radiologists.
 * **Catastrophic Failure:** Un-defended diagnostic models collapse under gradient attacks like PGD.
 * **Attack Vectors:** Digital interception on hospital PACS networks, sensor noise corruption, or malicious tampering.
* **Speaker Notes:** *"An attacker can add imperceptible noise to a normal chest X-ray, causing a CAD model to predict Pneumonia with 99% confidence. Conversely, a sick patient's X-ray could be manipulated to appear normal, delaying critical care."*

---

### Slide 5: Problem Statement
* **Core Problem:** Deep learning chest X-ray classifiers lack robust, integrated security against multi-attack adversarial perturbations.
* **Key Challenges in Existing Work:**
 1. *No Integrated Framework:* Existing defenses handle detection OR retraining OR denoising, but never an integrated pipeline.
 2. *Lack of Cross-Attack Generalization:* Defenses trained on single-step attacks (FGSM) fail against multi-step attacks (PGD) or boundary attacks (DeepFool).
 3. *Clean Accuracy Trade-off:* Adversarial retraining degrades diagnostic accuracy on clean clinical scans.
* **Speaker Notes:** *"Current research focuses on isolated defenses that either drop clean accuracy or fail against unseen attack types. AdvShield-Med addresses these three challenges simultaneously."*

---

### Slide 6: Objectives & Expected Outcomes
* **Project Objectives:**
 1. Build a baseline ResNet-18 diagnostic classifier for chest X-rays (Normal vs Pneumonia).
 2. Benchmark vulnerability under FGSM, PGD, and DeepFool attacks.
 3. Design a pre-inference **Gatekeeper CNN** for adversarial input detection.
 4. Develop a **U-Net Purifier Autoencoder** using combined (MSE + SSIM + VGG Perceptual) loss to repair perturbed images.
 5. Integrate components into an end-to-end pipeline with **Strict Rejection** and **Sanitize-First** operational modes.
 6. Package into a clinical web application (FastAPI backend + React frontend).
* **Speaker Notes:** *"Our objectives cover the full pipeline from baseline diagnostic modeling to adversarial detection, purification, pipeline integration, and web deployment."*

---

### Slide 7: Project Scope & Boundaries
* **In-Scope:**
 * Dataset: Kermany Pediatric Chest X-Ray (~5,856 images, 70/15/15 stratified split).
 * Architecture: ResNet-18 backbone (transfer learning).
 * Attacks: FGSM, PGD (multi-epsilon), DeepFool via `torchattacks`.
 * Defenses: Gatekeeper CNN + U-Net Purifier + Adversarially-Trained Backup Model.
 * Full-Stack App: FastAPI REST API + React 19 UI + PostgreSQL database.
* **Out-of-Scope:**
 * Integration with live hospital PACS hardware.
 * Federated learning across multi-center hospital networks.
 * Black-box transfer attacks without model access.
* **Speaker Notes:** *"We strictly bound our scope to binary chest X-ray classification on the Kermany dataset using standardized white-box attack benchmarks, backed by a functional web interface."*

---

### Slide 8: Literature Review Overview (15 Recent Papers: 2020-2026)
* **Themes Covered:**
 * *Vulnerability Analysis (2020-2023):* Chen et al. [2020], Ma et al. [2021], Hirano et al. [2021], Rahman et al. [2022], Vazirani et al. [2023].
 * *Detection Methods (2021-2024):* Park et al. [2021], Liu et al. [2022], Zhang et al. [2023], Kumar et al. [2024].
 * *Purification & Denoising (2021-2024):* Yoon et al. [2021], Nie et al. [2022], Al-Makhlafi et al. [2023], Sharma et al. [2024].
 * *Multi-Layer Defenses (2021-2025):* Pham et al. [2021], Singh et al. [2025].
* **Speaker Notes:** *"Our literature review covers 15 peer-reviewed papers published between 2020 and 2026, categorized across attack analysis, detection filters, purification autoencoders, and multi-layer architectures."*

---

### Slide 9: Literature Comparison Matrix
* **Table Columns:** Author/Year | Focus | Model / Method | Key Finding | Main Limitation
* **Key Entries:**
 * *Chen et al. [2020]:* Vulnerability | FGSM/PGD on ResNet | Medical AI highly sensitive | No defense proposed
 * *Zhang et al. [2023]:* Detection | Wavelet frequency filter | Detects FGSM noise | Tested only on FGSM
 * *Kumar et al. [2024]:* Detection | Gatekeeper binary CNN | Preserves 100% clean acc | Reject-only; no repair
 * *Sharma et al. [2024]:* Purification | MSE + SSIM + VGG Loss | Prevents structural blur | No upstream detection
 * *Singh et al. [2025]:* Architecture | Multi-layer framework | Multi-layer needed for unseen attacks | Conceptual framework
* **Speaker Notes:** *"As highlighted in Table 2.1 of our report, existing works either focus purely on detection without repair, or purification without upstream detection."*

---

### Slide 10: Identified Research Gap
* **4 Critical Gaps Addressed by AdvShield-Med:**
 1. **Single-Point Defense Deficit:** Existing methods use *only* detection or *only* denoising. We combine Gatekeeper + Purifier + Backup Model.
 2. **Cross-Attack Generalization Gap:** Defenses are rarely tested on unseen attack types. We train on FGSM and test on PGD and DeepFool.
 3. **Clean Accuracy Degradation:** Adversarial retraining lowers clean accuracy. Our Gatekeeper wrapper preserves 100% clean throughput.
 4. **Deployment Gap:** Theoretical models lack clinical software interfaces. We build a full-stack REST API and UI.
* **Speaker Notes:** *"AdvShield-Med bridges these four gaps by providing a multi-layered, cross-attack evaluated framework delivered as a clinical web application."*

---

### Slide 11: Proposed Methodology, Two-Part Defense Framework
* **Visual Diagram:**
 * **Part A, Adversarial Detection (Gatekeeper Module):**
 `Input X-Ray` $\rightarrow$ `Gatekeeper ResNet-18` $\rightarrow$ `Clean / Adversarial Flag` $\rightarrow$ `Route or Reject`.
 * **Part B, Adversarial Purification & Recovery (Purifier Module):**
 `Input X-Ray` $\rightarrow$ `U-Net Denoising Autoencoder` $\rightarrow$ `Repaired Radiograph` $\rightarrow$ `Diagnostic ResNet-18`.
* **Speaker Notes:** *"Our framework is structured into two main parts: Part A for early detection of tampered inputs via a Gatekeeper CNN, and Part B for spatial image purification via a U-Net autoencoder."*

---

### Slide 12: System Architecture & Workflow Modes
* **System Workflow Diagram:**
 * **Mode 1 (Strict Rejection):** Pre-inference filtering for high-security auditing. Suspicious images blocked immediately.
 * **Mode 2 (Sanitize-First):** Universal purification before classification. Ensures continuous diagnostic output even under attack.
* **Backend Architecture:** FastAPI async engine + PyTorch inference runtime + PostgreSQL database container.
* **Speaker Notes:** *"The system offers two operational paradigms: Strict Rejection for high-security environments, and Sanitize-First for continuous clinical automated diagnosis."*

---

### Slide 13: Module Specifications & Tech Stack
* **Software Modules:**
 1. `data.py`, Kermany loader, 70/15/15 split, WeightedRandomSampler.
 2. `models.py`, ResNet-18 diagnostic & gatekeeper builders.
 3. `attacks.py`, FGSM, PGD, DeepFool wrappers via `torchattacks`.
 4. `purifier.py`, U-Net autoencoder with `CombinedLoss` (MSE + SSIM + VGG).
 5. `pipeline.py`, End-to-end `AdvShieldPipeline` orchestration engine.
 6. Backend & Frontend, FastAPI REST API + React 19 UI + PostgreSQL.
* **Tech Stack:** PyTorch 2.2, Torchattacks 3.5, Scikit-Learn, FastAPI, React 19, Tailwind CSS 4, Docker.
* **Speaker Notes:** *"Our codebase is fully modular, built on PyTorch and torchattacks for the ML core, with FastAPI and React 19 powering the deployment layer."*

---

### Slide 14: Feasibility, Risk Management & Project Timeline
* **Risk Register:**
 * *Risk 1 (Compute Constraints):* PGD attacks are slow $\rightarrow$ Pre-generate and cache `.pt` attack tensors.
 * *Risk 2 (Cross-Attack Generalization):* Gatekeeper misses iterative attacks $\rightarrow$ Integrate Purifier autoencoder fallback.
 * *Risk 3 (Purification Blur):* MSE loss blurs edges $\rightarrow$ Incorporate SSIM + VGG perceptual feature loss.
 * *Risk 4 (Class Imbalance):* Kermany dataset imbalance $\rightarrow$ Apply WeightedRandomSampler and weighted loss.
* **Timeline (Gantt Summary):**
 * July (R1): Problem definition & scope.
 * August (R2): Literature review, methodology & system design.
 * September (R3): Detection implementation & interim testing.
 * October (R4/R5): Purification, full-stack app integration, final presentation.
* **Speaker Notes:** *"We have identified four primary technical risks and established clear mitigations. Our project timeline maps directly across the five semester review milestones."*

---

### Slide 15: Conclusion & Q&A
* **Summary:**
 * Comprehensive domain analysis of medical AI vulnerabilities.
 * Literature review of 15 recent papers (2020-2026) establishing the research gap.
 * Robust two-part methodology (Detection + Purification) supporting dual operational modes.
 * Complete modular system design and full-stack software architecture.
* **Thank You!**
* **Open for Panel Questions and Feedback**
* **Speaker Notes:** *"Thank you for your time and guidance. We are now ready to take questions and feedback from the panel."*
