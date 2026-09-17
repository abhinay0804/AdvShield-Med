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
