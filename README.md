# 🧠 Benchmarking GANs, Diffusion Models, and Flow Matching for T1w-to-T2w MRI Translation

This repository contains the **code and pretrained models** used in our paper:

> **Benchmarking GANs, Diffusion Models, and Flow Matching for T1w-to-T2w MRI Translation**  
> Andrea Moschetto, Lemuel Puglisi, Alec Sargood, Pierluigi Dell’Acqua, Francesco Guarnera, Sebastiano Battiato, Daniele Ravì  
> [arXiv:2507.14575](https://arxiv.org/abs/2507.14575)

---

## 🧩 Overview

Magnetic Resonance Imaging (MRI) provides different contrasts like **T1-weighted (T1w)** and **T2-weighted (T2w)** images. Acquiring both increases scan time and cost. Our work explores **image-to-image translation (I2I)** methods to synthesize T2w images from T1w inputs, potentially reducing acquisition time and cost in clinical practice.

We benchmark **three families of generative models** for this task:
- ⚔️ **Pix2Pix (GAN-based)**
- 🌫️ **Diffusion Models**
- 🌊 **Flow Matching Models**

All methods use a **shared U-Net architecture** and are trained/evaluated on three public MRI datasets (IXI, HCP, CamCAN). We provide **quantitative, qualitative, and computational comparisons**.

---

## 🚀 Getting Started


### 1. Clone the repository
```bash
git clone https://github.com/AndreaMoschetto/medical-I2I-benchmark.git
cd medical-I2I-benchmark
```

### 2. Set up environment
We recommend using [MONAI](https://monai.io) with Python 3.10 and PyTorch ≥ 1.12.
```bash
conda env create -f environment.yml
conda activate medical-i2i
```
### 3. (Optional) Install project in editable mode

If you want to use the package as a local module:

```bash
pip install -e .
```

This will make the `t1t2converter` package available for import across the project.

---

## 📂 Project Structure

```
medical-I2I-benchmark/
├── scripts/                   # Training & inference scripts for each model
│   ├── pix2pix_t1t2.py
│   ├── diffusion_t2.py
│   └── ...
├── models/                   # Exported weights
├── src/
│   └── t1t2converter/         # Core library: models, datasets, utilities
│       ├── __init__.py
│       ├── datasets.py
│       ├── models.py
│       └── utils.py
├── .env                      # Local environment variables (not tracked)
├── .gitignore
├── environment.yml           # Conda environment definition
├── LICENSE
├── pyproject.toml            # Project/package metadata
└── README.md
```

---

## ⚙️ Configuration

Runtime paths (like data, checkpoints, and outputs) are managed through the `utils.py` file, which reads values from environment variables. You can define your own paths by creating a `.env` file in the root directory:

```dotenv
# .env (not committed to version control)
MEDICAL_I2I_DATAPATH=/your/data/path
MEDICAL_I2I_OUTPUT=/your/output/path
MEDICAL_I2I_CKPT=/your/checkpoints/path
```

A template is provided as `.env.example`.

If not specified, default paths are created under the project directory.

---
