# STW–T cGAN 🫀⚡  
Spatiotemporal–Wavelet Transformer cGAN for ECGI Data Augmentation

![license](https://img.shields.io/github/license/dmayorcatorres/STW-T_cgan)
![paper](https://img.shields.io/badge/article-Elsevier%20(BSPC)-blue)
![python](https://img.shields.io/badge/python-3.11-blue.svg)

> Official implementation of the paper  
> **“STW–T cGAN: Dual-Branch Spatiotemporal–Wavelet Transformer for ECGI Data Augmentation”**  
> submitted to *Biomedical Signal Processing and Control* (Elsevier).

High-density ECGI datasets rarely exceed a few dozen cases. This repository tackles that bottleneck with a dual-branch model that **generates** synthetic heart surface potentials (HSPMs) and **reconstructs** body-surface potential maps (BSPMs), while preserving electrophysiological and geometric consistency.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Training and Evaluation](#-training-and-evaluation)
- [Reproducing Paper Results](#-reproducing-paper-results)
- [Model Overview](#-model-overview)
- [Citation](#-citation)
- [License and Copyright](#-license-and-copyright)
- [Contact](#-contact)

---

## 🔍 Overview

Electrocardiographic imaging (ECGI) seeks to reconstruct high-density epicardial electrical activity (HSPMs) from body surface recordings (BSPMs). In practice, ECGI research is limited by:

- **Ill-posed inverse problems** that amplify noise,  
- **Severe data scarcity** of paired BSPM–HSPM datasets, and  
- **High experimental cost and ethical constraints** for invasive recordings.

STW–T cGAN is a **dual-branch conditional GAN** that addresses these challenges by:

- Generating physiologically coherent **BSPM–HSPM pairs**, and  
- Enforcing **multi-domain fidelity** (time, frequency, and geometry) through physics-aware losses and architectural constraints.

This repository contains the **exact codebase used in the paper**, including model definitions, training and evaluation pipelines, configuration files, and scripts to reproduce the reported experiments.

---

## 🧾 Key Contributions

This implementation corresponds to the following main contributions of the paper:

1. **Dual-branch Spatiotemporal–Wavelet generator (STW–T cGAN)**  
   A conditional GAN architecture that jointly models BSPM–HSPM pairs in:
   - the **spatiotemporal domain** (ST branch, Transformer-based), and  
   - the **time–frequency domain** (W branch, continuous wavelet transform),  
   fused through **Feature-wise Linear Modulation (FiLM)**.  
   This design increases the diversity and realism of synthetic ECGI data while respecting underlying electrophysiological constraints.

2. **Geometry-aware spatial encoding via Reverse Cuthill–McKee (RCM)**  
   RCM reorders epicardial and torso electrodes so that geometrically adjacent electrodes become neighbors in the 1D channel index. This:
   - reduces inter-electrode graph distances by up to ~37%, and  
   - enables anatomically consistent pattern learning using 1D convolutions and axial attention.

3. **Multi-domain fidelity and stabilization against mode collapse**  
   The training objective combines:
   - **Adversarial loss** (non-saturating cGAN),
   - **Amplitude loss** (L1),
   - **Morphology loss** (per-channel Pearson correlation), and
   - **Spectral loss** (cosine similarity of Fourier spectra),  
   together with **spectral normalization** and an **auxiliary BSPM regression head**.  
   These components mitigate typical cGAN pathologies such as **mode collapse** and **conditioning information loss**, leading to stable training and physiologically meaningful samples.

4. **Auxiliary BSPM regression head for biophysical consistency**  
   A secondary head attached to the discriminator reconstructs BSPMs from learned latent features, providing:
   - additional gradient signals that regularize the discriminator, and  
   - an implicit enforcement of the forward BSPM–HSPM relationship.  
   This auxiliary task improves both generation quality and downstream ECGI reconstruction.

5. **Comprehensive evaluation on ex vivo and in vivo datasets**  
   The model is validated on:
   - **Bordeaux ex vivo** data (porcine), and  
   - **Auckland in vivo** data (porcine)  
   from the EDGAR archive, showing improvements in:
   - RMSE,  
   - Pearson correlation coefficient (PCC),  
   - spectral cosine similarity (SC), and  
   - maximum temporal cross-correlation,  
   compared to classical and deep-learning baselines.

---

## 📁 Project Structure

```text
STW-T_cgan/
├── src/
│   ├── models/           # Generator, discriminator, fusion blocks (ST, W, FiLM)
│   ├── data/             # Datasets, loaders, RCM utilities, transforms
│   ├── losses/           # Multi-domain fidelity losses, adversarial losses
│   ├── train.py          # Main training loop
│   └── evaluate.py       # Evaluation and metrics
├── configs/              # YAML configs (Auckland, Bordeaux, ablations)
├── scripts/              # Preprocessing, plotting, helper scripts
├── notebooks/            # Step-by-step demos and sanity checks
├── results/              # Trained weights, metrics, and figures (created after runs)
├── environment.yml       # Conda environment specification
└── README.md
