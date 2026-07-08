---
title: Cross-Modal Latent Fusion Playground
emoji: 🎭
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Dual-encoder sketch+RGB latent fusion (MSc, Surrey 2024)
---

# Cross-Modal Latent Fusion — Interactive Playground

Live demo of the runnable components of
[priyankak17/cross-modal-latent-fusion](https://github.com/priyankak17/cross-modal-latent-fusion):
"High-resolution Facial Image Synthesis from Low-resolution Images and Forensic Sketches"
(Priyanka Kamila, MSc CVRML, University of Surrey, 2024).

**Tabs**
1. **Degradation simulator** — `experiments/low_light_experiment.py` (low-light + resolution
   degradation), with an exact dissertation-parameters mode and L1/L2/PSNR/SSIM/LPIPS metrics.
2. **Sketch generator** — `scripts/sketch_gen.py` (Sobel-based sketch modality).
3. **Latent mixing playground** — the six W+ mixing strategies of dissertation §5 on proxy
   18×512 latents, including a bug-fixed `model/mixer.py`, with per-layer modality attribution.

**Honest scope:** the trained dual-encoder pSp checkpoints are not distributed with the repo,
so photorealistic StyleGAN2 decoding is out of scope for this CPU Space. Everything shown is
computed live and was validated quantitatively before publishing (see the "About & repo audit" tab).
