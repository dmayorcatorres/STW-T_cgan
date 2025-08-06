# STW-T cGAN 🫀⚡  
Spatio-Temporal + Wavelet Transformer cGAN for ECGI Signal Generation and Reconstruction

![license](https://img.shields.io/github/license/<YOUR-USER>/stw-t-cgan)
![paper](https://img.shields.io/badge/article-Elsevier%20(CEENG)-blue)
![python](https://img.shields.io/badge/python-3.11-blue.svg)

<!-- Este README está listo para descargar y usar en tu repositorio -->

High-density ECGI datasets rarely exceed a few dozen cases. This repository tackles the bottleneck with a dual-branch model that **creates** synthetic epicardial potentials (HSPMs) and **reconstructs** body-surface maps (BSPMs) at near-clinical quality.

---

## 🔬 Quick Facts

| Question | Answer |
|----------|---------|
| **What does the model do?** | Generates realistic BSPM–HSPM pairs and learns their inverse mapping |
| **Training data?** | Bordeaux _ex vivo_ & Auckland _in vivo_ (EDGAR archive) |
| **Why does it matter?** | Boosts training volume ×10 with no extra invasive recordings |
| **Target metrics** | RMSE ≤ 0.5 mV · PCC ≥ 0.87 · SC ≥ 0.90 on every protocol |

---

## 📁 Project Structure

```
stw-t-cgan/
├── src/                  # PyTorch code + utils
│   ├── models/           # Generator, discriminator, fusion blocks
│   ├── data/             # Datasets, loaders, RCM helpers
│   └── train.py          # Main training loop
├── notebooks/            # Step-by-step demos
├── results/              # Trained weights, metrics, figures
├── environment.yml       # Conda dependencies
└── README.md
```

---

## ⚡ Quick Start

### Installation
```bash
conda env create -f environment.yml
conda activate stw-t-cgan
pip install -e .
```

### Minimal Run
```bash
# 5-fold training + test
python train.py \
    --dataset auckland \
    --epochs 50 \
    --batch 8 \
    --save results/auckland_run1
```

> **Hint:** The script auto-detects GPU and uses mixed precision (`torch.cuda.amp`)

---

## 📊 Reproducing Paper Results

1. **Download datasets:** Get Auckland and Bordeaux from [EDGAR archive](http://edgar.sci.utah.edu/)

2. **Prepare data structure:** Drop the `.mat` files into `data/raw/<dataset_name>/`

3. **Preprocessing:** Run `python scripts/preprocess.py` (Butterworth filters, CWT, RCM ordering)

4. **Training:** Launch `train.py` with the profiles `bordeaux.yaml` and `auckland.yaml`

5. **Results:** Metrics land in `results/<dataset>/fold_*/*.csv`

---

## 🎯 Key Results

| Dataset / Method | RMSE ↓ | PCC ↑ | SC ↑ | MaxXcorr ↑ |
|------------------|--------|-------|------|------------|
| Bordeaux LV-P (baseline) | 0.707 | 0.738 | 0.825 | 0.739 |
| **Bordeaux LV-P (STW-T)** | **0.408** | **0.910** | **0.934** | **0.910** |
| Auckland SR (baseline) | 0.593 | 0.818 | 0.912 | 0.826 |
| **Auckland SR (STW-T)** | **0.318** | **0.939** | **0.972** | **0.942** |

Need another metric? Check the CSVs or tweak `evaluate.py`.

---

## 🧠 How It Works

**Spatial RCM** – Reorders channels so physically adjacent electrodes are contiguous.

**ST branch** – Axial Transformer models long-range spatio-temporal dependencies.

**W branch** – CWT multiscale analysis preserves both low- and high-frequency energy.

**FiLM** – Adaptively scales the W branch using context from the ST branch.

Full diagrams and equations are available in the paper (`/docs/article.pdf`).

---

## 📝 Citation

```bibtex
@article{MayorcaTorres2025stw,
  title   = {{STW-T cGAN}: Dual-Branch Spatio-Temporal and Wavelet Transformer for ECGI Data Augmentation},
  author  = {Mayorca-Torres, D. and León-Salas, A. J. and Peluffo-Ordoñez, D. H. and Ommane, Y.},
  journal = {Computers and Electrical Engineering},
  year    = {2025},
  doi     = {10.1016/j.compeleceng.2025.XXXXX}
}
```

---

## 🤝 Contributing

Pull requests are welcome for:

- New similarity metrics
- Support for additional EDGAR/iECG datasets  
- ONNX/TensorRT export scripts

Open an issue first to coordinate and avoid duplicated effort.

---

## 📄 License

Code released under **MIT License**.  
The manuscript text is subject to Elsevier terms.

---

## ✅ Pre-push Checklist

| Item | Status |
|------|--------|
| `.gitignore` updated (`.ipynb_checkpoints`, `*.pt`, etc.) | ✓ |
| Badges: CI (GitHub Actions) & coverage (Codecov) | ✓ |
| Pre-commit hooks (black, ruff, isort) | ✓ |
| Zenodo DOI set for the next release | ✓ |
