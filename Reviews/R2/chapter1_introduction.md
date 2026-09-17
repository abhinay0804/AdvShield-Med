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
