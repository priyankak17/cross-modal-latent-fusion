"""
Cross-Modal Latent Fusion — Interactive Playground
====================================================
Demo Space for: "High-resolution Facial Image Synthesis from Low-resolution
Images and Forensic Sketches" (Priyanka Kamila, MSc, University of Surrey, 2024)
Repo: https://github.com/priyankak17/cross-modal-latent-fusion

What runs live here (CPU, no checkpoints needed):
  1. Degradation Simulator  -> experiments/low_light_experiment.py
  2. Sketch Generator       -> scripts/sketch_gen.py (Sobel pipeline)
  3. Latent Mixing Playground -> model/mixer.py (bug-fixed) + the six
     mixing strategies of dissertation §5, applied to proxy W+ latents.

What does NOT run here: photorealistic StyleGAN2 decoding. The trained
dual-encoder pSp checkpoints are not distributed with the repository and
the decoder requires a GPU. The latent tab therefore operates on clearly
labelled *proxy* W+ latents (18x512) derived deterministically from the
input images, so the *behaviour* of every mixing strategy is real and
inspectable even though no face is synthesised.
"""

import io
import cv2
import numpy as np
import torch
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

# Optional perceptual metric (works on Spaces; degrades gracefully offline)
try:
    import lpips as lpips_lib

    _LPIPS = lpips_lib.LPIPS(net="alex")
    _LPIPS.eval()
except Exception:
    _LPIPS = None

N_LAYERS, LATENT_DIM = 18, 512  # StyleGAN2 W+ space at 1024px, as used by pSp

# ----------------------------------------------------------------------------
# 1) Degradation pipeline — faithful to experiments/low_light_experiment.py
# ----------------------------------------------------------------------------

def apply_low_light_effect(img_bgr, factor=0.3, sat_factor=1.0):
    """Repo implementation reduces only V. Dissertation §4.3.3 additionally
    reduces saturation by 0.5 — exposed here as `sat_factor`."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] *= factor
    hsv[..., 1] *= sat_factor
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def degrade_resolution(img_bgr, scale=0.5, nearest=False):
    """Repo default: scale=0.5, bilinear. Dissertation: 30% + nearest-neighbour."""
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    w = max(1, int(img_bgr.shape[1] * scale))
    h = max(1, int(img_bgr.shape[0] * scale))
    low = cv2.resize(img_bgr, (w, h), interpolation=interp)
    return cv2.resize(low, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=interp)


def image_metrics(a_rgb, b_rgb):
    """L1 / L2 / PSNR / SSIM (+ LPIPS if available) — evaluation style of
    scripts/calc_losses_on_images.py and dissertation §5."""
    a = a_rgb.astype(np.float32) / 255.0
    b = b_rgb.astype(np.float32) / 255.0
    l1 = float(np.abs(a - b).mean())
    l2 = float(((a - b) ** 2).mean())
    psnr = float("inf") if l2 == 0 else float(10 * np.log10(1.0 / l2))
    gray_a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)
    ssim = float(ssim_fn(gray_a, gray_b))
    rows = [["L1", f"{l1:.4f}"], ["L2 (MSE)", f"{l2:.4f}"],
            ["PSNR (dB)", f"{psnr:.2f}"], ["SSIM", f"{ssim:.4f}"]]
    if _LPIPS is not None:
        with torch.no_grad():
            ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
            rows.append(["LPIPS (alex)", f"{float(_LPIPS(ta, tb)):.4f}"])
    return rows


def run_degradation(img, brightness, saturation, scale, nearest):
    if img is None:
        return None, [], "Upload a face image to begin."
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = apply_low_light_effect(bgr, factor=brightness, sat_factor=saturation)
    out = degrade_resolution(out, scale=scale, nearest=nearest)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    v0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2].mean()
    v1 = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., 2].mean()
    note = (f"Mean brightness (HSV·V): {v0:.1f} → {v1:.1f} "
            f"(measured ratio {v1 / max(v0, 1e-6):.3f} vs. requested factor {brightness}). "
            "This degraded image is what Encoder B (low-res RGB) would receive.")
    return out_rgb, image_metrics(img, out_rgb), note


# ----------------------------------------------------------------------------
# 2) Sketch pipeline — faithful to scripts/sketch_gen.py
# ----------------------------------------------------------------------------

def sobel(img):
    x = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    y = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    return cv2.bitwise_or(x, y)


def make_sketch(img_rgb, blur_k=3, edge_weight=0.75):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur_k = int(blur_k) | 1  # force odd
    frame = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    edg = cv2.addWeighted(sobel(frame), edge_weight, sobel(255 - frame), edge_weight, 0)
    return 255 - edg


def run_sketch(img, blur_k, edge_weight):
    if img is None:
        return None, "Upload a face image to begin."
    sk = make_sketch(img, blur_k, edge_weight)
    note = (f"White background: {(sk > 200).mean() * 100:.1f}% of pixels · "
            f"edge pixels (<128): {(sk < 128).mean() * 100:.2f}%. "
            "This sketch is what Encoder A (sketch-to-face) would receive.")
    return sk, note


# ----------------------------------------------------------------------------
# 3) Latent mixing — model/mixer.py (fixed) + the six strategies of §5
# ----------------------------------------------------------------------------

def proxy_wplus(img_rgb, seed=0):
    """Deterministic proxy W+ latent (18x512) from an image.

    Layer k is built from image statistics at a resolution that grows with k,
    mirroring the coarse→fine semantics of StyleGAN's W+ layers. This is NOT
    a trained encoder — it exists so mixing strategies can be demonstrated
    and measured without the (undistributed) pSp checkpoints.
    """
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(4096, LATENT_DIM, generator=g) / np.sqrt(4096)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    layers = []
    for k in range(N_LAYERS):
        res = int(np.clip(4 * 2 ** (k // 2), 4, 64))
        small = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA).flatten()
        feat = np.zeros(4096, np.float32)
        feat[: small.size] = small - small.mean()
        v = torch.from_numpy(feat) @ proj
        layers.append(v / (v.norm() + 1e-8))
    return torch.stack(layers)  # (18, 512)


def slerp(a, b, t):
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    omega = torch.acos((a_n * b_n).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7))
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


def mix_latents(sketch_w, rgb_w, strategy, alpha, coarse_cut):
    """The six strategies of dissertation §5.0.1. sketch_w carries structure,
    rgb_w carries texture/colour. Returns (mixed, per-layer source share of
    sketch in [0,1] for the layer strip)."""
    c = int(coarse_cut)
    if strategy == "Weighted Average Mixing":
        out = alpha * sketch_w + (1 - alpha) * rgb_w
        share = np.full(N_LAYERS, alpha)
    elif strategy == "Feature Addition & Splitting":
        s = sketch_w + rgb_w  # enhance shared features...
        # ...then split back to source scale per layer
        scale = 0.5 * (sketch_w.norm(dim=-1, keepdim=True) + rgb_w.norm(dim=-1, keepdim=True))
        out = s / (s.norm(dim=-1, keepdim=True) + 1e-8) * scale
        share = np.full(N_LAYERS, 0.5)
    elif strategy == "Selective Feature Combination":
        # This is the corrected model/mixer.py: coarse layers (structure)
        # from the sketch encoder, fine layers (texture) from the RGB encoder.
        out = torch.cat([sketch_w[:c], rgb_w[c:]], dim=0)
        share = np.array([1.0] * c + [0.0] * (N_LAYERS - c))
    elif strategy == "Residual Learning":
        out = rgb_w + alpha * (sketch_w - rgb_w) * 0.6  # residual on RGB base
        share = np.full(N_LAYERS, alpha * 0.6)
    elif strategy == "Feature Scaling":
        gate = torch.sigmoid((sketch_w.abs() - rgb_w.abs()) * 8.0)  # channel-wise
        out = gate * sketch_w + (1 - gate) * rgb_w
        share = gate.mean(dim=-1).numpy()
    else:  # Latent Space Interpolation (dissertation's best performer)
        out = slerp(sketch_w, rgb_w, 1 - alpha)
        share = np.full(N_LAYERS, alpha)
    return out, share


def cosine_to(a, b):
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return float((a_n * b_n).sum(-1).mean())


def plot_latents(sketch_w, rgb_w, mixed, share, strategy):
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 7.2),
                             gridspec_kw={"height_ratios": [3, 3, 3, 1]})
    fig.patch.set_facecolor("#151a21")
    names = ["Sketch encoder W+ (proxy)", "RGB encoder W+ (proxy)",
             f"Mixed W+ — {strategy}"]
    for ax, w, name in zip(axes[:3], [sketch_w, rgb_w, mixed], names):
        ax.imshow(w.numpy(), aspect="auto", cmap="magma", vmin=-0.12, vmax=0.12)
        ax.set_ylabel("layer", color="#c8d2dc", fontsize=8)
        ax.set_title(name, color="#e8eef4", fontsize=10, loc="left", family="monospace")
        ax.tick_params(colors="#8a97a5", labelsize=7)
    strip = np.tile(share, (1, 1))
    axes[3].imshow(strip, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[3].set_yticks([])
    axes[3].set_xticks(range(N_LAYERS))
    axes[3].set_xticklabels(range(N_LAYERS), fontsize=7, color="#8a97a5")
    axes[3].set_title("per-layer sketch contribution  (red = sketch/structure, blue = RGB/texture)",
                      color="#e8eef4", fontsize=9, loc="left", family="monospace")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


DISSERTATION_TABLE = [
    ["Latent Space Interpolation", "0.1466", "0.0566", "0.3639", "best on all three metrics"],
    ["(range across all 6 strategies)", "0.1466–0.4085", "0.0566–0.2943", "0.3639–0.7433",
     "high variance → mixing instability"],
]


def run_mixing(img, strategy, alpha, coarse_cut, brightness, scale):
    if img is None:
        return None, None, None, [], "Upload a face image to begin."
    # Build both modality inputs from the photo (the dissertation pipeline)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    degraded = cv2.cvtColor(
        degrade_resolution(apply_low_light_effect(bgr, brightness), scale),
        cv2.COLOR_BGR2RGB)
    sketch = make_sketch(img)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    sketch_w = proxy_wplus(sketch_rgb)
    rgb_w = proxy_wplus(degraded)
    mixed, share = mix_latents(sketch_w, rgb_w, strategy, alpha, coarse_cut)

    heat = plot_latents(sketch_w, rgb_w, mixed, share, strategy)
    stats = [
        ["cos(mixed, sketch W+)", f"{cosine_to(mixed, sketch_w):.4f}"],
        ["cos(mixed, RGB W+)", f"{cosine_to(mixed, rgb_w):.4f}"],
        ["‖mixed‖ / mean(‖inputs‖)",
         f"{float(mixed.norm() / (0.5 * (sketch_w.norm() + rgb_w.norm()))):.4f}"],
        ["layers from sketch (share > 0.5)", f"{int((share > 0.5).sum())} / {N_LAYERS}"],
    ]
    note = ("Latents shown are proxy W+ codes (18×512) — the mixing math is exactly "
            "the dissertation's; photorealistic decoding needs the trained pSp "
            "checkpoints + StyleGAN2 GPU decoder, which the repo does not ship.")
    return sketch, degraded, heat, stats, note


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

CSS = """
.gradio-container {font-family: 'Inter', 'Segoe UI', sans-serif;}
#hdr h1 {font-size: 1.55rem; letter-spacing: -0.01em; margin-bottom: 0.15rem;}
#hdr p  {color: var(--body-text-color-subdued); margin-top: 0;}
.mono, .mono * {font-family: 'JetBrains Mono', ui-monospace, monospace !important;}
#warn {border-left: 4px solid #e0a83c; padding: 8px 12px; background: rgba(224,168,60,.08); border-radius: 6px;}
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)

_GR6 = int(gr.__version__.split(".")[0]) >= 6
_style_kwargs = {"theme": theme, "css": CSS}

with gr.Blocks(title="Cross-Modal Latent Fusion — Playground",
               **({} if _GR6 else _style_kwargs)) as demo:
    with gr.Column(elem_id="hdr"):
        gr.Markdown(
            "# Cross-Modal Latent Fusion — Playground\n"
            "Dual-encoder face synthesis from **forensic sketches** (structure, amber) + "
            "**low-light/low-res photos** (texture, cyan) · MSc dissertation, University of Surrey 2024 · "
            "[repo](https://github.com/priyankak17/cross-modal-latent-fusion)")
        gr.Markdown(
            "**Scope note:** the trained dual-encoder checkpoints are not distributed with the "
            "repository, so this Space runs the repo's *runnable* components live (degradation, "
            "sketch generation, latent mixing math + metrics) and does not synthesise photorealistic faces.",
            elem_id="warn")

    with gr.Tab("1 · Degradation simulator"):
        gr.Markdown("Runs `experiments/low_light_experiment.py`: HSV brightness reduction + "
                    "down/up-scale pixelation. Toggle **dissertation mode** for the exact §4.3.3 "
                    "parameters (V×0.3, S×0.5, 30% nearest-neighbour).")
        with gr.Row():
            with gr.Column():
                d_in = gr.Image(label="Input face", type="numpy", height=320)
                d_bright = gr.Slider(0.05, 1.0, 0.3, step=0.05, label="Brightness factor (V)")
                d_sat = gr.Slider(0.1, 1.0, 1.0, step=0.05,
                                  label="Saturation factor (repo keeps 1.0; dissertation uses 0.5)")
                d_scale = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="Resolution scale")
                d_nn = gr.Checkbox(False, label="Nearest-neighbour interpolation (dissertation mode)")
                d_btn = gr.Button("Degrade", variant="primary")
            with gr.Column():
                d_out = gr.Image(label="Degraded output (Encoder B input)", height=320)
                d_tbl = gr.Dataframe(headers=["metric", "value"], label="Fidelity vs. original",
                                     elem_classes="mono")
                d_note = gr.Markdown()
        gr.Examples([["examples/face1.jpg"], ["examples/face2.jpg"]], inputs=[d_in])
        d_btn.click(run_degradation, [d_in, d_bright, d_sat, d_scale, d_nn], [d_out, d_tbl, d_note])

    with gr.Tab("2 · Sketch generator"):
        gr.Markdown("Runs `scripts/sketch_gen.py`: Gaussian blur → dual Sobel (image + inverse) → "
                    "weighted merge → invert. Produces the structural modality for Encoder A.")
        with gr.Row():
            with gr.Column():
                s_in = gr.Image(label="Input face", type="numpy", height=320)
                s_blur = gr.Slider(1, 9, 3, step=2, label="Gaussian blur kernel")
                s_w = gr.Slider(0.25, 1.5, 0.75, step=0.05, label="Edge weight")
                s_btn = gr.Button("Generate sketch", variant="primary")
            with gr.Column():
                s_out = gr.Image(label="Sketch (Encoder A input)", height=380)
                s_note = gr.Markdown()
        gr.Examples([["examples/face1.jpg"], ["examples/face2.jpg"]], inputs=[s_in])
        s_btn.click(run_sketch, [s_in, s_blur, s_w], [s_out, s_note])

    with gr.Tab("3 · Latent mixing playground"):
        gr.Markdown(
            "One photo → both modality inputs → proxy W+ latents (18×512) → one of the **six "
            "mixing strategies** from dissertation §5. The layer strip is the key readout: it shows "
            "which StyleGAN layers each modality controls (coarse 0–4 ≈ pose/structure, fine 5–17 ≈ "
            "texture/colour). *Selective Feature Combination* is the bug-fixed `model/mixer.py` — the "
            "committed version drops the RGB latents entirely.")
        with gr.Row():
            with gr.Column(scale=1):
                m_in = gr.Image(label="Input face", type="numpy", height=280)
                m_strategy = gr.Dropdown(
                    ["Weighted Average Mixing", "Feature Addition & Splitting",
                     "Selective Feature Combination", "Residual Learning",
                     "Feature Scaling", "Latent Space Interpolation"],
                    value="Selective Feature Combination", label="Mixing strategy (§5.0.1)")
                m_alpha = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="α — sketch weight")
                m_cut = gr.Slider(1, 17, 5, step=1,
                                  label="Coarse/fine cut layer (mixer.py uses 5)")
                m_bright = gr.Slider(0.05, 1.0, 0.3, step=0.05, label="Degradation: brightness")
                m_scale = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="Degradation: resolution")
                m_btn = gr.Button("Mix latents", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    m_sk = gr.Image(label="Modality A — sketch (structure)", height=200)
                    m_dg = gr.Image(label="Modality B — degraded RGB (texture)", height=200)
                m_heat = gr.Image(label="W+ latents & per-layer modality attribution")
                m_tbl = gr.Dataframe(headers=["property", "value"],
                                     label="Mixing diagnostics", elem_classes="mono")
                m_note = gr.Markdown()
        gr.Examples([["examples/face1.jpg"], ["examples/face2.jpg"]], inputs=[m_in])
        m_btn.click(run_mixing, [m_in, m_strategy, m_alpha, m_cut, m_bright, m_scale],
                    [m_sk, m_dg, m_heat, m_tbl, m_note])
        gr.Markdown("#### Reference results reported in the dissertation (§5.0.2–5.0.3)")
        gr.Dataframe(value=DISSERTATION_TABLE,
                     headers=["strategy", "L1", "L2", "LPIPS", "note"],
                     interactive=False, elem_classes="mono")

    with gr.Tab("About & repo audit"):
        gr.Markdown("""
### The research
A **dual-encoder pSp architecture**: one encoder maps a *forensic sketch* to StyleGAN2's W+ space,
a second maps a *low-light / low-resolution photo* to the same space; their style vectors are mixed
and decoded by a frozen, pre-trained StyleGAN2 generator into a high-resolution face. Six latent
mixing strategies were compared with L1 / L2 / LPIPS against ground truth; **latent-space
interpolation** performed best (L1 0.1466 · L2 0.0566 · LPIPS 0.3639), while high variance across
samples showed the mixing process is unstable — the dissertation's central finding is that aggregate
metrics conceal these failure modes.

### What this audit found in the repository
| Finding | Detail |
|---|---|
| Framework files missing | `inference*.py`, `style_mixing.py`, `train*.py` import `models/psp.py`, `configs/`, `options/`, `datasets/`, `training/coach_dualencoders.py` — none are in the repo (they belong to the upstream [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) codebase; the dual-encoder coach itself is absent). |
| No checkpoints | The trained CelebA-HQ dual encoders and StyleGAN2 weights are not distributed → full generation cannot be reproduced. |
| `model/mixer.py` bug | Loads both latent banks but concatenates `mix_2[:, :5]` with `mix_2[:, 5:]` — i.e. the output is *identically the sketch latents*; `mix_1` (RGB) is never used. Fixed here as `cat([sketch[:5], rgb[5:]])`. |
| `scripts/sketch_gen.py` | Requires `pip install torchfile` although the `.t7` model loading is commented out (stale import). |
| Code vs. dissertation drift | Dissertation §4.3.3: V×0.3, **S×0.5**, 30% **nearest-neighbour**. Repo `low_light_experiment.py`: V×0.3 only, 50% bilinear. Both variants are exposed in Tab 1. |
| README paths | References `models/mixer.py` (folder is `model/`) and `assets/architecture.png` (absent). |

### Validation performed before publishing this Space
Degradation: measured V-ratio 0.296 vs. requested 0.30 ✓ · Laplacian variance 15.3 → 2.7 (pixelation) ✓.
Sketch: white background 93%+, dark edge fraction 0.5% ✓. Mixing: selective combination verified
layer-exact (0–4 ≡ sketch, 5–17 ≡ RGB), interpolation endpoints recover each source ✓.

*Ethics: intended for research/education on multimodal latent fusion. Forensic face synthesis has
serious misuse and bias risks (the dissertation discusses this in §1.1.1); do not use outputs to
identify real people.*
""")

if __name__ == "__main__":
    demo.launch(**(_style_kwargs if _GR6 else {}))
