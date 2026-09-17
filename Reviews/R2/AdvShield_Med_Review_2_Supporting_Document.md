# AdvShield-Med: A Multi-Attack Adversarial Detection and Defense Framework for Robust Deep Learning-Based Chest X-Ray Diagnosis

**School of Computer Science and Engineering (SCOPE), VIT Chennai Campus**  
**Course & Code:** BCSE497J - Project - I (Fall Semester 2026-2027)  
**Milestone:** Review 2 (Panel Review, Evaluation Weight: 20 Marks)  
**Document Type:** Unified Supporting Technical Document (Chapters 1-3)

---

# Abstract

Recent studies show that deep learning models used for computer-aided diagnosis (CAD) for radiology have strong baseline diagnostic accuracy but are very susceptible to adversarial attacks. Additions of noise on the pixel level can cause a dramatic change in the diagnostic prediction, and this can be done with confidence, which poses serious risks to the safety of patients in clinical applications. This supporting document will outline the full technical rationale, extensive literature review of 15 recent peer-reviewed studies over the last 6 years (2020-2026), and detailed system methodology for the development of AdvShield-Med: a multi-layered adversarial detection and defense framework for chest radiograph diagnosis. Two components of AdvShield-Med were implemented: (Part A) An upstream Gatekeeper CNN trained for adversarial input detection during the inference process and (Part B) A U-Net spatial denoising autoencoder, trained using a composite loss function (MSE, SSIM, and VGG perceptual loss) to reconstruct clean radiographs from attacked inputs. There are two flexible operating modes: Strict Rejection Mode for high security auditing and Sanitize-First Mode for continual automatic diagnosis. Detected by its critical research gaps in the field of cross-attack generalization and clean accuracy preservation, AdvShield-Med is built with the framework of PyTorch, Torchattacks, FastAPI, PostgreSQL and React 19, without compromising the diagnostic backbone.

---

# Chapter 1: Introduction

## 1.1 Background and Motivation

In recent 10 years, Convolutional Neural Networks (CNNs) have been introduced to the clinical radiology workflow. For common chest radiograph screening tasks, DL models are comparable or superior to radiologists in detecting common diseases like pneumonia [Rajpurkar et al., 2017]. Chest X-ray is the most widely performed imaging test in the world, and is the least expensive and quickest [Wang et al., 2017]. As a result, the ability to categorize chest radiographs as normal or pathological is a commonly used testbed for clinical AI systems [Kermany et al., 2018].

But these models have a huge security flaw. Trained networks can make incorrect predictions with high probability despite being fed small, algorithmically-chosen variations in the pixels that make up an image [Szegedy et al., 2014]. This was shown by Goodfellow et al. [2014] who were able to change the results of the classification with a single step along the gradient sign. This was later extended by Madry et al. [2018] who proposed an iterative attack, known as Projected Gradient Descent (PGD), that maximizes the perturbations within a specified bounded vicinity of the original image. As demonstrated by Moosavi-Dezfooli et al. [2016], making such minimal perturbations can cause predictions to be incorrect.

This is especially true in the healthcare sector, where it can directly impact patient care. For healthcare, the implications of this vulnerability are immediate and significant. If the pneumonia-positive X-ray is changed to a normal diagnosis by an adversarial perturbation, treatment will be delayed. If it results in a false positive, then the patient's healthy scans are unnecessarily followed by procedures. In consumer applications, computer vision fails do not pose an immediate threat to patients whereas in radiology, failures can have a direct impact on the patient. Medical AI needs to be robust against adversarial noise for its deployment in hospital.

Adversarial training and input preprocessing are currently the two types of defenses. In adversarial training [Madry et al., 2018] images are perturbed and added to the training data directly. This will make the training process more robust to the attack it is subjected to. However, it always decreases the accuracy of the clean, non-disturbed images [Tsipras et al., 2019]. The trade-off doesn't work in hospitals, because nearly every incoming scan is uncontaminated. Input preprocessing techniques such as spatial filtering, JPEG compression or feature squeezing [Xu et al., 2018] attempt to clean up the perturbations prior to classification. They are prone to failing when there are multiple-step or adaptive attacks. Most importantly, the researchers typically only test defenses against one attack algorithm. It is still unclear whether a defense that is trained on FGSM can resist unseen attacks such as PGD or DeepFool.

These problems resulted in AdvShield-Med. This project aims to create a modular framework for defense of chest X-ray data for diagnosis while preserving diagnostic accuracy for benign data, while at the same time detecting and repairing adversarial data.

## 1.2 Problem Statement

Deep learning models achieve good performance on clean X-ray images of chest X-rays, but fail in the presence of adversarial perturbations. Even though the diagnosis CNN models are well-trained, the changes made to the image are so small and so subtle that the models are tricked to provide the wrong diagnosis with a high success rate. This means that patient safety is jeopardized and there is the inability to deploy a safe image network in PACS.

In present defense literature there are three fundamental issues:

1. **No standard pipeline:** There is no standard pipeline for the detection and purification of products. The existing studies tackle the detection, image repair and strong classification process individually. To the best of our knowledge, no single deployment system contains all three components: pre-inference detection, autoencoder purification and diagnostic classification.
2. **Unverified cross-attack generalization:** Most published defenses are judged on the specific attack algorithm that they were trained on. They do not explore if a defense that rejects FGSM can detect and clean unseen iterative attacks, such as PGD or DeepFool.
3. **Lack of accuracy in clean images:** Clean accuracy decreases with retraining diagnostic models with adversarial images. In clinical practice, where the majority of cases are benign, it is not acceptable to have an accuracy that drops off.

AdvShield-Med includes a Gatekeeper detector and a U-Net Purifier around an unmodified diagnostic model to overcome these three challenges.

## 1.3 Objectives

The project has six specific technical objectives:

1. **Train baseline diagnostic model:** Train a ResNet-18 on chest X-rays (Normal vs. Pneumonia) to achieve a baseline of accuracy in a clean setting.
2. **Benchmark adversarial vulnerability:** Numerically measure the loss of diagnostic accuracy for the model under FGSM, multi-epsilon PGD and DeepFool attacks.
3. **Learn to detect an adversary (Gatekeeper):** Fit a binary ResNet-18 network to classify whether or not adversarial noise is present prior to images being fed into the diagnostic network.
4. **Create Image Purifier:** Use a combined loss function (MSE, SSIM, VGG feature loss) to train a U-Net autoencoder to remove adversarial noise and preserve lung structures.
5. **Put in place an integrated defense pipeline:** Implement Gatekeeper, Purifier and an adversarially-trained backup model in a single runtime system, that runs in either of two modes - strict rejection and sanitize-first.
6. **Evaluate cross-attack generalization:** Only train against FGSM perturbations and test on unseen PGD and DeepFool attacks.

## 1.4 Scope

- **Dataset:** Kermany Pediatric Chest X-Ray (~5,856 images) dataset, with Normal and Pneumonia classes.
- **Attacks:** White-box FGSM (epsilon in {0.01, 0.03, 0.06}), multi-step PGD (40 steps with the same epsilon values), and DeepFool (50 steps).
- **Defense System:** Pretrained ResNet-18 diagnostic classifier, ResNet-18 Gatekeeper detector, U-Net Purifier autoencoder and an adversarially-trained backup classifier.
- **Deployment Model:** A FastAPI REST backend with weights, with a React/TypeScript web dashboard for demonstrations.
- **Outside the scope:** Direct hospital PACS/DICOM hardware integration, Federated learning across multiple health systems, black-box transfer attacks (without access to the gradients of the model) and Multi-label datasets such as NIH ChestX-ray14.

## 1.5 Expected Outcomes

1. A baseline diagnostic ResNet-18 with a minimum clean accuracy of 85% on a set of Kermany test radiographs.
2. Evidence that the baseline diagnostic accuracy drops close to 0% when attacked by FGSM, PGD and DeepFool.
3. A Gatekeeper detector with an F1 score of ≥ 0.85 on FGSM and good detection rates on unseen PGD attacks.
4. An Adversarial Noise Removal and Diagnostic Accuracy Restoration Purifier autoencoder without spatial blurring.
5. A whole pipeline built using the Python script and working both for rejection and purification.
6. The image upload and security screening features are implemented in a working full-stack web dashboard that features diagnostic outputs.

## 1.6 Organization of the Report

This report is organized by topic.

- Chapter 2 (Literature Review) explores 15 recent papers (2020-2026) in four categories: medical challenges for AI vulnerabilities, frequency/intermediate detection, autoencoder purification and multi-layered defense frameworks. It provides an overview of current gaps in research.
- Chapter 3 (Methodology and System Design) displays the details of the two-part defense architecture, the specifications of the modules, the loss function math, the software pipeline logic, and the tech stack details.
- Chapter 4 (Implementation and Results) covers the experimental setup, baseline training, attack benchmarks, generalization testing with the gatekeeper, purifier recovery performance and integration with a web app.
- Chapter 5 (Conclusion and Future Work) presents project results, an analysis of the limitations of the existing system and future research directions.

---

# Chapter 2: Literature Review

## 2.1 Medical AI Vulnerabilities

CNNs are effective in various clinical diagnostic applications, such as chest radiograph analysis, mammography and histopathology. But the thing is, these models have one big drawback - they are susceptible to noise from other sources. The outputs of a model are distorted by small, localized pixel perturbations, without being noticed by human clinicians.

The following papers published in the last 7 years (i.e., 2020-2026) have been reviewed: The Studies selected are: Adversarial Vulnerability in Radiology, Detection Algorithms, Autoencoder Purification and Multi-Layer Defense Architects. The aim is to describe existing approaches, their technical constraints and identify the research gap to be filled by AdvShield-Med.

## 2.2 Review of Recent Literature (2020-2026)

### 2.2.1 Weakness of Medical Imaging

1. **Chen et al. [2020] - Adversarial Susceptibility Across Medical Modalities (IEEE TMI)**  
The researchers tested the standard gradient attacks (FGSM, PGD, Carlini & Wagner (C&W)) on ResNet-50 and DenseNet-121 classifiers on chest X-rays, dermoscopy and fundus images. Their tests indicate that the accuracy of models trained on medical imaging data degrades at a higher rate than models trained on natural images (CIFAR-10, ImageNet) when attacked. The authors correlated this sensitivity with the low intra-class variability in the visual features of clinical X-rays and the high spatial redundancy.

2. **Ma et al. [2021] - Feature Space Behavior Under Low Magnitude Perturbations (Pattern Recognition)**  
Ma et al. studied feature maps on the models of an X-rayed chest in a disturbed state. They discovered that when they encounter adversarial noise, intermediate feature representations are carried across decision boundaries, yet with only small spatial alterations. In models of pneumonia and tuberculosis, the low perturbation budgets ($\epsilon \le 0.01$) reduced diagnostic accuracy to less than 15%.

3. **Hirano et al. [2021] - Universal Adversarial Perturbations in Radiology (IEEE Access)**  
This paper demonstrates the effectiveness of UAP for a real-world task in radiology. Hirano et al. experimented on UAPs on a chest X-ray and CT classifier, which are image-agnostic. An additional static noise vector, which was added to all test images, led to false-negative misclassifications in more than 85% of the samples. Their results showed that they could not possibly have to create perturbations specific to each image to break clinical classifiers.

4. **Pham et al. [2021] - Shift of Grad-CAM Feature Map in Presence of Noise (Computer Methods and Programs in Biomedicine)**  
Pham et al. experimented with ResNet-18, DenseNet-121 and EfficientNet on the Kermany chest X-ray dataset, presenting an accuracy of over 95% on a clean baseline. They were able to trace the model attention with Grad-CAM heat maps. Without noise, the model could pay attention to the lungs; with noise, it could instead pay attention to irrelevant peripheral borders, which is why it was not able to be accurate in the presence of noise.

5. **Rahman et al. [2022] - Multi-Scale Iterative Attacks on Chest Radiographs (Medical Image Analysis)**  
Rahman et al. tested several multi-scale iterative attacks against chest X-ray models. They showed that 40 steps of PGD attacks are able to overcome simple image resizing and image normalization. Multi-step iterative attacks also successfully outperformed the single-step FGSM attack, thus ensuring PGD is a useful baseline for evaluating defenses.

6. **Vazirani et al. [2023] - Threat Models for Clinical AI Deployments (Nature Machine Intelligence)**  
Vazirani et al. classified the security threat vectors in radiology departments of hospitals. They examined the interference of DICOM images on PACS networks with the interference of scanner hardware artifacts and direct alteration of the image pixels. They determined that network-level image interception is a very feasible risk and that it would need to be pre-inferenced to be validated before the image was sent to a diagnostic algorithm.

### 2.2.2 Detection-Based Defenses

7. **Park et al. [2021] - Frequency-Domain Detection of Gradient Attacks (IEEE TPAMI)**  
Park et al. proposed detecting adversarial inputs via spatial-frequency analysis. Computing Discrete Cosine Transform (DCT) coefficients revealed that gradient-based attacks leave high-frequency signatures. A classifier trained on these spectral features identified adversarial inputs in natural images. However, compression artifacts common in medical PACS storage led to high false-positive rates.

8. **Liu et al. [2022] - Intermediate Feature Representation Distillation (MICCAI)**  
Liu et al. built an adversarial detector for medical image classifiers using intermediate network features. Rather than inspecting raw pixels, their method extracts feature maps from residual blocks in the diagnostic network. An auxiliary classifier uses these maps to flag attacks, obtaining F1 scores above 0.90 on known attack types. Its detection rate dropped on small-step iterative attacks.

9. **Zhang et al. [2023] - Wavelet Decomposition for Chest X-Ray Input Validation (Computers in Biology and Medicine)**  
Zhang et al. combined high-pass wavelet decomposition with spatial edge statistics to isolate adversarial noise in chest X-rays. Their detector separated artificial noise from pathological lung opacities, achieving high detection rates on FGSM images. But because they tested only on FGSM, performance against iterative attacks like PGD remained unverified.

10. **Kumar et al. [2024] - Pre-Inference Gatekeeper Security Modules (IEEE JBHI)**  
Kumar et al. designed a binary Gatekeeper network positioned in front of an unmodified diagnostic model. The Gatekeeper evaluates inputs first, rejecting suspicious images before diagnosis. This design preserved 100% of the diagnostic model's clean accuracy. However, because it operated solely as a rejection filter, flagged patients received no automated diagnosis.

### 2.2.3 Autoencoders and Image Purification

11. **Yoon et al. [2021] - Convolutional Autoencoders for Medical Image Denoising (Computers in Biology and Medicine)**  
Yoon et al. evaluated autoencoder preprocessing to clean adversarial noise from radiographs. They trained a Convolutional Autoencoder (CAE) with Mean Squared Error (MSE) loss to reconstruct clean images from FGSM inputs. While it removed low-amplitude noise, the MSE loss caused spatial blurring along high-contrast anatomical edges (ribs and cardiac borders), slightly lowering clean diagnostic accuracy.

12. **Nie et al. [2022] - Diffusion-Based Purification (DiffPure) (ICML)**  
Nie et al. introduced DiffPure, using forward and reverse diffusion steps to scrub adversarial noise. DiffPure removed multiple attack types without needing attack-specific training. However, iterative diffusion sampling takes several seconds per image, making it too slow for high-throughput radiology workflows.

13. **Al-Makhlafi et al. [2023] - U-Net Skip Connections for Radiograph Denoising (Biomedical Signal Processing and Control)**  
Al-Makhlafi et al. built a U-Net autoencoder to clean adversarial radiographs. Skip connections between encoder and decoder layers preserved spatial anatomical boundaries while the bottleneck filtered out noise. The U-Net cleaned FGSM images effectively, but relying on MSE loss alone limited recovery on multi-step PGD attacks.

14. **Sharma et al. [2024] - Multi-Term Loss Formulations for Radiograph Reconstruction (Artificial Intelligence in Medicine)**  
Sharma et al. improved autoencoder purification by combining MSE loss with Structural Similarity (SSIM) loss and VGG-16 perceptual feature loss. This multi-term loss prevented edge blurring and retained lung opacity details better than MSE alone, recovering diagnostic accuracy across perturbed images.

### 2.2.4 Multi-Layer Defense Integration

15. **Singh et al. [2025] - Cross-Attack Generalization in Safety-Critical Medical AI (IEEE TNNLS)**  
Singh et al. evaluated cross-attack generalization in medical AI defenses. They showed that defenses trained on single-step attacks (FGSM) often fail against multi-step attacks (PGD) or boundary attacks (DeepFool). They recommended multi-layered defenses - combining input detection, autoencoder purification, and robust backup models - to maintain protection across varied threat types.

## 2.3 Methodological Comparison of Reviewed Literature

Table 2.1 compares the fifteen reviewed papers across their target focus, models, datasets, main results, and limitations.

### Table 2.1: Comparison of Recent Literature (2020-2026)

| Study | Primary Focus | Model / Attack | Dataset | Key Result | Main Limitation |
|:---|:---|:---|:---|:---|:---|
| Chen et al. [2020] | Vulnerability | FGSM, PGD, C&W on ResNet | Chest X-Ray, Dermoscopy | Medical models drop accuracy faster than natural vision models. | Diagnostic evaluation only; no defense proposed. |
| Ma et al. [2021] | Vulnerability | Feature space shift under low $\epsilon$ | Chest X-Ray, CT | $\epsilon \le 0.01$ perturbations drop accuracy below 15%. | No detection or repair modules developed. |
| Hirano et al. [2021] | Threat Model | Universal Adversarial Perturbations | Chest X-Ray, CT | A single static UAP vector forces >85% false negatives. | Evaluated only static UAP vectors, not dynamic iterative attacks. |
| Pham et al. [2021] | Interpretability | ResNet-18 + Grad-CAM heatmaps | Kermany Chest X-Ray | >95% clean accuracy, but noise shifts attention to margins. | Did not test a defense to fix heatmap shifts. |
| Rahman et al. [2022] | Vulnerability | 40-step multi-scale PGD | Chest X-Ray | 40-step PGD drops accuracy to near 0%, bypassing simple resizing. | Attack analysis only; no defense mechanism built. |
| Vazirani et al. [2023] | Threat Taxonomy | DICOM network interception model | Clinical Radiology | Defined DICOM PACS network interception risks. | Conceptual threat framework without code evaluation. |
| Park et al. [2021] | Detection | High-frequency DCT classifier | Natural Images | Spectral analysis isolates gradient attack signatures. | High false-positive rate on compressed medical DICOMs. |
| Liu et al. [2022] | Detection | Intermediate feature distillation | Medical Segmentation | F1 > 0.90 on known attacks using internal layer features. | Detection drops on small-step iterative attacks. |
| Zhang et al. [2023] | Detection | Wavelet Decomposition + Edge Stats | Chest X-Ray | High detection accuracy on FGSM images. | Evaluated only on FGSM; no PGD or DeepFool tests. |
| Kumar et al. [2024] | Detection | Pre-Inference Binary Gatekeeper | CAD Systems | Preserved 100% clean accuracy by filtering inputs. | Rejection-only design; rejected scans get no diagnosis. |
| Yoon et al. [2021] | Purification | Convolutional Autoencoder (MSE) | Chest X-Ray | Removes low-amplitude noise, but blurs anatomical edges. | MSE loss alone degrades clean anatomical detail. |
| Nie et al. [2022] | Purification | Diffusion Purification (DiffPure) | Natural Images | Removes multiple attack types without attack-specific training. | High inference latency (seconds per image) slows clinical workflow. |
| Al-Makhlafi et al. [2023] | Purification | U-Net with Skip Connections | Medical Images | Skip connections preserve spatial structure better than standard CAEs. | Tested mostly on single-step noise; MSE loss limits PGD recovery. |
| Sharma et al. [2024] | Purification | Autoencoder with MSE + SSIM + VGG Loss | Radiographs | Multi-term loss preserves lung anatomy and prevents blurring. | Evaluated as a standalone denoiser without upstream detection. |
| Singh et al. [2025] | Architecture | Multi-Layer Defense Analysis | Medical AI Systems | Single-layer defenses fail on unseen attacks; multi-layer required. | Conceptual blueprint without an integrated web software implementation. |

## 2.4 Limitations of Current Research

Current research has a number of limitations, as detailed below. These studies reveal three major gaps in the medical AI security literature:

1. **Model Retraining Accuracy Loss:** Retraining diagnostic networks on adversarial images (Chen et al. [2020], Ma et al. [2021]) improves the robustness to adversarial images, but reduces the accuracy on the clean images. The overall rate of baseline scans that are clean is very high, so a drop in overall diagnostic accuracy is not acceptable.
2. **Unverified Cross-Attack Generalization (UCAG):** Most detection and denoising papers use single-step attacks for testing (Zhang et al. [2023] and Al-Makhlafi et al. [2023] used FGSM as their attack type). Multi-step attacks (PGD) or boundary attacks (DeepFool), as Singh et al. [2025] noted, are attacks that defenses trained on a single-step noise are likely to fail against.
3. **Isolated Defense Components:** Published works either focus on making only detection decisions (Kumar et al. [2024]) and drop flagged images without giving a diagnosis, or denoise the image alone (Sharma et al. [2024]) which is a very costly operation because it requires heavy image reconstruction on each and every image, whether it is clean or attacked.

## 2.5 Research Gap Addressed by AdvShield-Med

AdvShield-Med addresses the following research gap: There is no literature in the medical AI field that incorporates all the following:

- **Layered Security Wrapper:** Wraps unmodified diagnostic model with an upstream Gatekeeper detector, a U-Net Purifier autoencoder and an adversarially-trained backup classifier.
- **Cross-Attack Generalization Testing:** Trains detection/purification modules on FGSM, and tests on unseen PGD and DeepFool attacks.
- **Flexible Operating Modes:** Can operate either in a Strict Rejection Mode for high security auditing or in a Sanitize-First Mode for continuous diagnostic output.
- **A Deployable Web Application:** Contains the whole defense pipeline with a FastAPI backend and a user interface done with React/TypeScript.

AdvShield-Med was created to address this unique need.

---

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

- **Clinical User Interface:** React 19 + TypeScript + Tailwind CSS 4 UI dashboard. Allows uploading of X-ray images via drag and drop, and real time security alerts (Clean vs Adversarial), confidence % and diagnosis.
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

---

# REFERENCES

1. [1] Al-Makhlafi et al. [2023]. U-Net-based denoising and purification against adversarial noise in medical imaging. *Biomedical Signal Processing and Control*, 85, 104920.
2. [2] Chen et al. [2020]. Systematic evaluation of adversarial vulnerability in medical image analysis. *IEEE Transactions on Medical Imaging*, 39(12), 4210-4221.
3. [3] Goodfellow, I. J., Shlens, J., & Szegedy, C. [2014]. Explaining and harnessing adversarial examples. *International Conference on Learning Representations (ICLR)*.
4. [4] Hirano, H., Minagi, A., & Takemoto, K. [2021]. Universal adversarial perturbations in medical image classification. *IEEE Access*, 9, 8956-8964.
5. [5] Kermany, D. S., et al. [2018]. Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.
6. [6] Kumar, R., et al. [2024]. Multi-stage gatekeeper networks for securing computer-aided diagnostic classifiers. *IEEE Journal of Biomedical and Health Informatics*, 28(3), 1420-1431.
7. [7] Liu, Y., et al. [2022]. Adversarial detection for medical image analysis via intermediate representation distillation. *MICCAI 2022*, LNCS 13431, 412-422.
8. [8] Ma, X., et al. [2021]. Understanding adversarial attacks on deep learning based medical image analysis systems. *Pattern Recognition*, 110, 107332.
9. [9] Madry, A., et al. [2018]. Towards deep learning models resistant to adversarial attacks. *International Conference on Learning Representations (ICLR)*.
10. [10] Moosavi-Dezfooli, S. M., Seyed-Mohsen, A., & Frossard, P. [2016]. DeepFool: a simple and accurate method to fool deep neural networks. *IEEE CVPR*, 2574-2582.
11. [11] Nie, W., et al. [2022]. Diffusion-based purification for robust image classification. *International Conference on Machine Learning (ICML)*.
12. [12] Park, J., et al. [2021]. Frequency-domain detection of adversarial examples in deep neural networks. *IEEE TPAMI*, 43(11), 3980-3993.
13. [13] Pham, T. Q., et al. [2021]. Interpretable and robust deep learning for pneumonia detection in chest radiographs. *Computer Methods and Programs in Biomedicine*, 208, 106277.
14. [14] Rahman, M. A., et al. [2022]. Vulnerability of chest X-ray AI diagnostic classifiers to multi-scale adversarial perturbations. *Medical Image Analysis*, 78, 102410.
15. [15] Rajpurkar, P., et al. [2017]. CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv preprint arXiv:1711.05225*.
16. [16] Sharma, A., et al. [2024]. Deep reconstruction networks for neutralizing adversarial perturbations in radiographs. *Artificial Intelligence in Medicine*, 147, 102740.
17. [17] Singh, A., et al. [2025]. Cross-attack generalization and defense frameworks in safety-critical medical AI systems. *IEEE Transactions on Neural Networks and Learning Systems*, 36(2), 890-903.
18. [18] Szegedy, C., et al. [2014]. Intriguing properties of neural networks. *International Conference on Learning Representations (ICLR)*.
19. [19] Tsipras, D., et al. [2019]. Robustness may be at odds with accuracy. *International Conference on Learning Representations (ICLR)*.
20. [20] Vazirani, A., et al. [2023]. Threat models and adversarial robustness of deep learning in radiology. *Nature Machine Intelligence*, 5(4), 340-350.
21. [21] Wang, X., et al. [2017]. ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. *IEEE CVPR*, 2097-2106.
22. [22] Xu, W., Evans, D., & Qi, Y. [2018]. Feature squeezing: Detecting adversarial examples in deep neural networks. *Network and Distributed System Symposium (NDSS)*.
23. [23] Yoon, H., et al. [2021]. Adversarial denoising with convolutional autoencoders for medical image protection. *Computers in Biology and Medicine*, 135, 104610.
24. [24] Zhang, K., et al. [2023]. Frequency-domain adversarial input detection for robust medical image analysis. *Computers in Biology and Medicine*, 158, 106820.
