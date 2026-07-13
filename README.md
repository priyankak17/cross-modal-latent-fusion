# Cross-Modal Latent Fusion for High-Resolution Facial Image Synthesis

**High-resolution facial image synthesis from low-resolution photographs and forensic sketches, via a dual-encoder pSp architecture over StyleGAN2's W+ latent space.**

MSc Dissertation · MSc Computer Vision, Robotics & Machine Learning · University of Surrey · September 2024
Supervised by Prof. Yi-Zhe Song and Subhadeep Koley

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Live Demo](https://img.shields.io/badge/🤗%20Demo-Playground-yellow.svg)](https://huggingface.co/spaces/pynk17/cross-modal-latent-fusion-playground)

---

## Overview

This project investigates a practical and under-explored problem in generative modelling: how to reconstruct a high-resolution, photorealistic face when the only available evidence is **degraded** — a low-light, low-resolution photograph — and **abstract** — a forensic-style sketch. Each modality is individually insufficient. A sketch carries structure (pose, facial geometry, proportions) but no texture or colour; a degraded photo carries colour and texture cues but little usable detail. The central research question is whether the two can be **fused in latent space** to recover more than either provides alone, and — more interestingly — *which modality a model learns to prioritise when the two disagree.*

The approach uses a **dual-encoder** design built on the pixel2style2pixel (pSp) framework. Each modality is encoded independently into StyleGAN2's extended **W+** latent space, the two sets of style vectors are combined by a latent mixing module, and a frozen, pre-trained StyleGAN2 generator decodes the fused code into a 1024×1024 image. Because both encoders project directly into W+, the method needs **no per-image optimisation** at inference — a notable efficiency advantage over optimisation-based GAN inversion.

> A key contribution is methodological: the work argues that aggregate reconstruction metrics can *conceal* failure modes, and designs an evaluation around that concern rather than around a single headline score.

---

## Method

### Dual-encoder architecture

```
Forensic sketch  ──►  Sketch Encoder (pSp)  ──►  W+  ┐
                                                     ├──►  Latent Mixing  ──►  StyleGAN2 (frozen)  ──►  1024² face
Low-light / low-res photo  ──►  RGB Encoder (pSp)  ──►  W+  ┘
```

- **Sketch encoder** — translates sparse line structure into W+ style vectors, handling line ambiguity and varying sketch detail. Contributes coarse structure.
- **RGB (low-quality) encoder** — a pSp-based encoder with a Feature Pyramid Network and Map2Style blocks, trained on CelebA-HQ, that maps degraded photographs into W+. Contributes texture and colour.
- **Frozen StyleGAN2 decoder** — a pre-trained generator used as a fixed decoder, so image quality benefits from large-scale pretraining while training cost stays low.

### Latent mixing strategies

Six strategies for combining the two W+ codes were designed and compared:

| # | Strategy | Idea |
|---|----------|------|
| 1 | Weighted Average Mixing (α) | Balanced linear blend of both codes |
| 2 | Feature Addition & Splitting | Add then re-split features to emphasise shared structure |
| 3 | Selective Feature Combination | Take coarse (structural) layers from the sketch, fine (texture) layers from the photo |
| 4 | Residual Learning | Preserve one modality as a base and add the residual of the other |
| 5 | Feature Scaling | Selectively amplify some feature channels while suppressing others |
| 6 | Latent Space Interpolation | Interpolate between the two latent vectors |

### Data preparation

Trained on **CelebA-HQ** (30,000 images at 1024², encoder trained at 512²); **FFHQ** used as reference. Paired degraded inputs were synthesised with a controlled pipeline: HSV brightness ×0.3 and saturation ×0.5 (low-light simulation), followed by 30% nearest-neighbour down/up-scaling (resolution/pixelation degradation), preserving a one-to-one correspondence with high-quality ground truth.

---

## Results

Evaluated against ground-truth RGB images using pixel-wise (L1, L2) and perceptual (LPIPS) metrics.

**Latent Space Interpolation** was the strongest strategy across all three metrics:

| Metric | Best (Latent Interpolation) | Range across all six strategies |
|--------|-----------------------------|---------------------------------|
| L1     | **0.1466**                  | 0.1466 – 0.4085 |
| L2     | **0.0566**                  | 0.0566 – 0.2943 |
| LPIPS  | **0.3639**                  | 0.3639 – 0.7433 |

**Key findings**

- In its best cases, latent fusion approaches the quality of pure-RGB reconstruction — evidence that structure and texture *can* be combined constructively.
- Performance is **highly variable** across samples, indicating instability in the mixing process (likely non-linear interactions in W+ and sensitivity to input characteristics).
- LPIPS stays comparatively high even when pixel-wise error is low, underscoring that perceptual realism is not captured by pixel metrics alone.
- The distribution of outcomes is skewed toward degradation rather than enhancement — a concrete instance of a model satisfying a specified objective while deviating from intended behaviour.

---

## Repository structure

```
cross-modal-latent-fusion/
├── dissertation/
│   └── Priyanka_Kamila_MSc_Dissertation.pdf   # full 52-page dissertation
├── model/
│   └── mixer.py                               # latent mixing
├── experiments/
│   └── low_light_experiment.py                # degradation pipeline
├── scripts/                                   # sketch generation, inference, training utilities
│   ├── sketch_gen.py
│   └── ...
├── utils/
│   └── visualise.py                           # output visualisation
└── README.md
```

> **Note on reproducibility.** This repository documents the research and shares the mixing, degradation, and evaluation code. The inference/training scripts depend on the upstream [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) framework, and the trained dual-encoder checkpoints are not distributed, so end-to-end photorealistic generation is not reproducible from this repository alone. An interactive demo of the *runnable* components degradation, sketch generation, and all six latent-mixing strategies is available below.

---

## Interactive demo

A live playground on Hugging Face Spaces demonstrates the degradation simulator, the sketch generator, and all six latent-mixing strategies (with per-layer modality attribution and L1/L2/PSNR/SSIM/LPIPS readouts):

**🤗 [huggingface.co/spaces/pynk17/cross-modal-latent-fusion-playground](https://huggingface.co/spaces/pynk17/cross-modal-latent-fusion-playground)**

---

## Why this work

The methodology of designing experiments to understand *how* a model resolves competing objectives, not merely *whether* it scores well, connects directly to broader questions in interpretability, robustness evaluation, and the gap between a specified training objective and learned behaviour. The recurring observation that the model optimises a measured proxy while diverging from the intended goal is a small, empirical encounter with the objective-misspecification problems that motivate much current work in reliable and safe machine learning.

**Ethics.** Facial reconstruction from sketches and degraded imagery has clear forensic value but equally clear potential for misuse and demographic bias. This work is intended for research and education; generated outputs should not be treated as reliable identifications of real individuals. These considerations are discussed further in the dissertation.

---

## Citation

```bibtex
@mastersthesis{kamila2024crossmodal,
  title   = {High-resolution Facial Image Synthesis from Low-resolution Images and Forensic Sketches},
  author  = {Kamila, Priyanka},
  school  = {University of Surrey},
  year    = {2024},
  program = {MSc Computer Vision, Robotics and Machine Learning},
  note    = {Supervised by Prof. Yi-Zhe Song and Subhadeep Koley}
}
```

---

## Author

**Priyanka Kamila**
MSc Computer Vision, Robotics & Machine Learning — University of Surrey, 2024

<!-- Update these links before publishing -->
[LinkedIn](https://linkedin.com/in/your-handle) · [Email](mailto:you@example.com)

---

*Built on [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) and [StyleGAN2](https://github.com/NVlabs/stylegan2). Grateful to Prof. Yi-Zhe Song and Subhadeep Koley for supervision.*
