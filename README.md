# Cross-Modal Latent Fusion for Multimodal Generative Modeling

> MSc Dissertation · University of Surrey · Computer Vision, Robotics & Machine Learning · 2024

[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)

-

## Overview

This repository contains the dissertation, model architecture, and experiments for my MSc research on **controllable multimodal image synthesis through cross-modal latent fusion**.

The central question: *when a generative model is given two competing input signals and a composite training objective, which signal does it learn to prioritise — and why?*

Rather than treating this as a pure generation task, the research was designed as an empirical investigation into **how models resolve competing objectives**, with implications for interpretability, controllability, and the gap between specified loss functions and learned behaviour.

---

## Architecture

![Architecture Diagram](./assets/architecture.png)

A **dual-encoder** architecture processes two input modalities in parallel, projecting each into a shared latent space before fusion via `models/mixer.py`:

```
Sketch Input  ──► Encoder A ──►┐
                                ├──► Latent Mixing Module ──► Decoder ──► Synthesised Image
RGB Input     ──► Encoder B ──►┘
```

Three latent mixing strategies were designed, implemented, and compared:

| Strategy | Description | Key Property |
|---|---|---|
| **Feature Interpolation** | Weighted linear combination of latent vectors | Smooth, continuous blending |
| **Residual Fusion** | Additive residual connection between encoded representations | Preserves modality-specific structure |
| **Selective Channel Gating** | Learned gating mask selects channels per modality | Sparse, disentangled mixing |

---


Key findings across mixing strategies:

- **Selective Channel Gating** produced the strongest representation disentanglement, allowing each modality to contribute selectively rather than blending uniformly
- **Residual Fusion** was most stable during training but showed evidence of modality dominance under certain initialisation conditions
- **Feature Interpolation** produced the smoothest outputs but the least controllable latent structure
- Models optimising composite loss functions frequently satisfied the metric while deviating from intended generative behaviour — motivating more granular, ablation-driven evaluation design

---

## Repository Structure

```
cross-modal-latent-fusion/
│
├── README.md
├── dissertation/
│   └── Priyanka_Kamila_MSc_Dissertation.pdf   # Full MSc dissertation
│
├── models/
│   └── mixer.py                                # Dual-encoder + latent mixing strategies
│
├── experiments/
│   └── low_light_experiment.py                 # Evaluation under low-light conditions
│
├── utils/
│   └── visualise.py                            # Latent space and output visualisation
│
├── scripts/
   └── ...                                     # Training and evaluation scripts

```

---

## Experiments

### Low-Light Generalisation (`experiments/low_light_experiment.py`)

One of the key evaluation settings investigates model robustness under **low-light input conditions** that is a distribution shift that stress-tests whether the model has learned meaningful latent representations or is exploiting surface-level statistics in the training data.

This experiment is directly motivated by the dissertation's core finding: aggregate metrics can conceal failure modes. A model may achieve low reconstruction loss under standard conditions while its latent representations degrade significantly under mild distribution shift.

---

## Evaluation Approach

A core contribution of this work is the **evaluation methodology**. Aggregate metrics alone are insufficient to understand *how* a model is mixing; they tell you whether the output looks good, not why.

The framework combines:

- **Pixel-wise metrics** — L1, L2 reconstruction loss for fidelity
- **Perceptual metrics** — LPIPS for perceptual realism beyond pixel distance
- **Ablation studies** — systematic isolation of architectural decisions to attribute performance to specific design choices
- **Distribution shift testing** — evaluation under low-light conditions to probe generalisation and representation robustness

---

## Relevance to Broader Research

The methodological approach developed here, designing experiments to understand *how* a model resolves competing objectives, rather than just *whether* it performs well, transfers directly to problems in:

- **Interpretability**: Understanding which inputs a model attends to under a composite objective
- **Alignment research**: Empirically studying the gap between specified training objectives and learned behaviour
- **Robustness evaluation**: Building evaluation frameworks that expose failure modes concealed by aggregate metrics

The recurring finding that models optimise for measured proxies rather than intended objectives, which is achieving low loss while deviating from desired behaviour, is a concrete encounter with the kind of objective misspecification problem that motivates much of the current empirical AI safety literature.

---

## Reading the Dissertation

The full dissertation is available in [`dissertation/Priyanka_Kamila_MSc_Dissertation.pdf`](./dissertation/Priyanka_Kamila_MSc_Dissertation.pdf).

It covers:
- Motivation and related work in multimodal generative modeling
- Architecture design and mixing strategy formulation
- Training methodology and loss function design
- Experimental results and ablation analysis
- Discussion of failure modes and directions for future work

---

## Citation

```bibtex
@mastersthesis{kamila2024latent,
  title     = {Cross-Modal Latent Fusion for Multimodal Generative Modeling},
  author    = {Kamila, Priyanka},
  school    = {University of Surrey},
  year      = {2024},
  program   = {MSc Computer Vision, Robotics and Machine Learning}
}
```

---

## Author

**Priyanka Kamila** · [LinkedIn]([https://linkedin.com/in/priyankakamila]) · [Email](kamilapriyanka17@gmail.com)

MSc Computer Vision, Robotics & Machine Learning — University of Surrey, 2024


**Priyanka Kamila** · [LinkedIn](https://linkedin.com/in/yourprofile) · [Email](mailto:your@email.com)

MSc Computer Vision, Robotics & Machine Learning — University of Surrey, 2024
