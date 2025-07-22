# 🧠 Benchmarking GANs, Diffusion Models, and Flow Matching for T1w-to-T2w MRI Translation

This repository contains the **code and pretrained models** used in our paper:

> **Benchmarking GANs, Diffusion Models, and Flow Matching for T1w-to-T2w MRI Translation**  
> Andrea Moschetto, Lemuel Puglisi, Alec Sargood, Pierluigi Dell’Acqua, Francesco Guarnera, Sebastiano Battiato, Daniele Ravì  
> [arXiv:2507.14575](https://arxiv.org/abs/2507.14575)

---

Magnetic Resonance Imaging (MRI) provides different contrasts like **T1-weighted (T1w)** and **T2-weighted (T2w)** images. Acquiring both increases scan time and cost. Our work explores **image-to-image translation (I2I)** methods to synthesize T2w images from T1w inputs, potentially reducing acquisition time and cost in clinical practice.

We benchmark **three families of generative models** for this task:
- ⚔️ **Pix2Pix (GAN-based)**
- 🌫️ **Diffusion Models**
- 🌊 **Flow Matching Models**

All methods use a **shared U-Net architecture** and are trained/evaluated on three public MRI datasets (IXI, HCP, CamCAN). We provide **quantitative, qualitative, and computational comparisons**.

---