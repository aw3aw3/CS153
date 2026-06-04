# CS153 — Final Project Implementation

Computer-vision pipeline that ingests a photograph of a petrographic thin
section (cross-polarized light), **segments** individual mineral grains,
**classifies** each grain by mineral, and reports the thin section's
**modal mineralogy** (mineral distribution).

```
image  →  SAM segmentation  →  per-grain crops  →  mineral classifier
       →  count% / area% summary  +  per-grain uncertainty  +  visual outputs
```

## Quick start (dashboard)

```powershell
conda activate cs153
pip install -r requirements.txt
streamlit run app.py
```

A browser opens at `http://localhost:8501`. Click **➕ New sample**, name it, and
upload its **series of stage-rotation photos** (same field of view at different
stage angles). Hit **Analyze sample** and you get: an **interactive mineral map
you can hover** (each grain shows its predicted mineral, confidence and
uncertainty), an uncertainty map, the modal mineralogy (area % / count %), a
per-grain table, and a downloadable JSON report. The rotations are fused into one
stronger prediction. Each analyzed sample is listed in the sidebar to revisit.
No other setup needed (weights auto-download on first run).

**Target rock type: granite**, scoped to 8 minerals: quartz, plagioclase,
microcline, orthoclase, biotite, muscovite, hornblende, zircon. The classifier
never predicts outside this set; grains it isn't sure about surface through the
uncertainty layer, and non-mineral regions through background detection.

## The classifier — a trained CNN (default)

The default backend (**`finetuned`**) is a **ResNet-18 convolutional neural
network**, ImageNet-pretrained and **fine-tuned via transfer learning on the
[MUMDMC2025](https://www.nature.com/articles/s41597-025-05879-9) granite
thin-section dataset** (Menoufia University, CC-BY 4.0). Details:

- **Training data:** cross-polarized photomicrograph crops of granite minerals,
  imaged across many microscope stage-rotation angles.
- **Classes:** quartz, plagioclase, orthoclase (K-feldspar), biotite, hornblende
  — the granite minerals with labeled training data available.
- **Method:** transfer learning (Adam, cross-entropy with **label smoothing** for
  calibrated confidences), with rotation/flip/crop/colour-jitter augmentation
  **plus grain-mask augmentation** — training crops are masked onto the same grey
  background the segmentation pipeline produces, so the model trains under the
  conditions it sees at inference (no train/inference domain gap). **100 %
  held-out validation accuracy**. Inference uses **test-time augmentation**
  (averaging several rotated views per grain) for robustness.
- **Shipped:** the trained weights (`checkpoints/mineral_cnn.pt`) are included in
  the repo, so the app works out of the box — no training step needed.

The model is self-describing (the checkpoint carries its own label set), and the
classifier is **pluggable** — all backends share one `classify(crops)` interface:
- **`finetuned`** (default) — the trained ResNet-18 above.
- **`clip`** — CLIP zero-shot fallback; no training, works on any rock type, but
  lower accuracy. Uses the `granite` mineral-description prompts.
- **`claude`** — Claude vision API (optional; needs `ANTHROPIC_API_KEY`).

## Layout

```
src/
  segmentation.py   SAM grain segmentation
  crops.py          per-grain crop extraction
  minerals.py       mineral vocabulary + zero-shot prompt templates
  classify.py       pluggable classifiers (CLIP / Claude / fine-tuned CNN)
  cnn.py            shared CNN architecture + transforms (train & inference)
  uncertainty.py    test-time augmentation + entropy/margin/agreement
  stats.py          modal-mineralogy aggregation (count% / area%)
  viz.py            mineral-colored overlay + distribution chart
  pipeline.py       analyze_thin_section() — the importable entrypoint
app.py              Streamlit dashboard (upload → analyze → results)
segment.py          CLI: segmentation only
analyze.py          CLI: full segment + classify + summarize
train_classifier.py CLI: fine-tune the CNN backend on labeled crops
data/               Input images (gitignored)
outputs/            Per-run outputs (gitignored)
checkpoints/        Trained mineral CNN (mineral_cnn.pt, shipped); SAM weights (gitignored)
```

## Storage (large files live off the SSD)

Heavy assets are kept on a secondary drive to spare the (small) C: SSD and avoid
OneDrive sync churn. Persistent user env vars route them there:

| Env var | Points to | Holds |
|---|---|---|
| `CS153_CHECKPOINT_DIR` | `D:\ml_cache\checkpoints` | SAM weights (~358 MB) |
| `HF_HOME` | `D:\ml_cache\huggingface` | CLIP weights |
| `TORCH_HOME` | `D:\ml_cache\torch` | torchvision/ResNet weights |

The MUMDMC2025 dataset lives at `D:\datasets\mumdmc2025\`. The fine-tuned model
checkpoint (`checkpoints/mineral_cnn.pt`, ~43 MB) stays in the repo tree.

## Setup

Python 3.10+. Heavy install (~3 GB: torch + SAM + CLIP).

```powershell
conda activate cs153
pip install -r requirements.txt
```

**GPU strongly recommended.** SAM on CPU is ~6× slower. Install a CUDA build of
torch to match your driver, e.g.:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Weights download on first run: SAM checkpoint (~375 MB) into `checkpoints/`,
CLIP weights (~600 MB) into the HuggingFace cache.

## Usage

### Full analysis (segment + classify + modal mineralogy)

```powershell
python analyze.py data\my_thin_section.jpg
python analyze.py img.jpg --minerals ultramafic --min-confidence 0.15
python analyze.py img.jpg --backend finetuned     # trained CNN
python analyze.py img.jpg --backend claude        # needs ANTHROPIC_API_KEY
```

Outputs land in `outputs/<image_stem>_analysis/`:
- `mineral_overlay.png` — grains colored by predicted mineral
- `uncertainty_overlay.png` — grains shaded green (certain) → red (uncertain)
- `distribution.png` — area% / count% bar chart (colors match the overlay)
- `summary.json` — modal summary + per-grain mineral, confidence, **entropy,
  margin, agreement**, and top-3 scores

Key flags:
- `--backend {finetuned,clip,claude}` — classifier (default `finetuned`)
- `--checkpoint PATH` — trained-model weights for `--backend finetuned`
- `--minerals {granite,default,ultramafic}` — candidate mineral set (default
  `granite`; clip/claude only). Edit `GRANITE_MINERALS` in `src/minerals.py` to
  refine the diagnostic descriptions.
- `--model {vit_b,vit_l,vit_h}` — SAM backbone
- `--min-area N`, `--long-edge N`, `--points-per-side N` — segmentation tuning

### Uncertainty

Every run reports per-grain uncertainty, and the pipeline can relabel shaky
grains as `uncertain` instead of guessing:

- **`--tta` / `--no-tta`** — test-time augmentation (default **on**): classifies
  ~5 rotated/flipped views per grain and averages them (rotation is the
  petrographically meaningful augmentation — minerals change under stage
  rotation). Yields a per-grain `agreement` (view stability) and more robust,
  better-calibrated probabilities.
- Per-grain signals in `summary.json`: `entropy` (0–1, spread of the
  distribution), `margin` (top1−top2), `agreement` (TTA view consensus).
- Relabel thresholds (a grain → `uncertain` if it trips **any**):
  `--min-confidence F`, `--max-entropy F` (0–1), `--min-agreement F` (0–1).

```powershell
# Flag grains the model isn't sure about (high entropy or unstable under rotation)
python analyze.py img.jpg --backend finetuned --max-entropy 0.6 --min-agreement 0.6
```

The model is deliberately trained with label smoothing so confidences aren't
wildly overconfident; on out-of-domain crops entropy stays high, so thresholds
push them to `uncertain` rather than a confident wrong label.

### Training the fine-tuned classifier (MUMDMC2025)

Fine-tune the `finetuned` backend on labeled thin-section crops. Defaults target
the [MUMDMC2025 dataset](https://www.nature.com/articles/s41597-025-05879-9)
(Menoufia University, CC-BY 4.0): photomicrographs of 5 granite minerals
(biotite, hornblende, plagioclase, orthoclase, quartz) under PPL/XPL across 72
stage-rotation angles. Labels are inferred from each file's path. The repo
already ships a trained `checkpoints/mineral_cnn.pt`; retrain only to change it.

```powershell
python train_classifier.py --data-root D:\datasets\mumdmc2025\extracted\MUMDMC2025_DataSet\Cropped_Images
python train_classifier.py --data-root DIR --split-by group --polarization xpl
```

Writes `checkpoints/mineral_cnn.pt` (self-describing: carries its own labels,
model name, input size). Key flags: `--split-by {group,image}` (group = hold out
whole crystals = honest eval), `--polarization {both,xpl,ppl}`, `--epochs`,
`--model {resnet18,resnet34,resnet50}`, `--max-per-class`.

### Multi-view fusion (stronger predictions)

Upload several images of the **same field of view** (e.g. different stage
rotations, focus, or exposure) and the tool fuses them into one stronger result:

- Grains are segmented on the first (reference) image; the others are aligned to
  it (ECC affine) and the **same masks** are reused.
- Each grain is classified in every view and the probability distributions are
  **averaged** — more views → steadier estimate, and views that *disagree* push
  the grain toward `uncertain` (per-grain `agreement` = cross-view consensus).

In the dashboard, uploading 2+ images shows a **"Combine as multiple views"**
option (default). Programmatically:

```python
from src.pipeline import analyze_thin_section_multi
result = analyze_thin_section_multi(["rot0.jpg", "rot30.jpg", "rot60.jpg"],
                                    classifier=clf, align=True)
```

Example: a quartz crystal where a *single* view ranked quartz 3rd (27 % area)
came out **quartz-dominant (59 %)** after fusing three rotation views.

### Background / non-grain handling

SAM also segments things that aren't minerals — background, mounting epoxy,
holes, image edges. These are detected **photometrically** (not by the
classifier): under cross-polarized light, true background is *isotropic* — dark
and unsaturated — while minerals show interference colors. Such grains are
labeled `non-grain` and **excluded from the modal mineralogy** (reported
separately). This is backend-agnostic and won't misclassify colorful grains.

- `--keep-non-grain` — keep them in the percentages (default: exclude).
- `--bg-dark-frac F` — a grain is `non-grain` if ≥ F of its pixels are
  dark+unsaturated (default 0.6; lower = stricter).

### Programmatic entrypoint

A future upload UI should build the classifier **once** and reuse it:

```python
from src.classify import build_classifier
from src.pipeline import analyze_thin_section

clf = build_classifier("finetuned")            # load weights once, reuse per upload
result = analyze_thin_section(
    "img.jpg", classifier=clf,
    tta=True, max_entropy=0.6, min_agreement=0.6,   # uncertainty handling
)

print(result.summary.minerals[0].mineral)      # dominant mineral
print(f"{result.n_uncertain}/{result.n_grains} grains uncertain")
for s in result.summary.minerals:
    print(s.mineral, f"{s.area_frac*100:.1f}% area  entropy={s.mean_entropy:.2f}")
```

### Segmentation only

```powershell
python segment.py data\my_thin_section.jpg
```

## AI Usage Disclosure

AI tools were used during development of this project for:

- **Wiring together the pipeline** — connecting the segmentation, classification,
  and aggregation stages into a cohesive end-to-end flow.
- **Searching for data** — locating a suitable labeled thin-section dataset
  (MUMDMC2025) for training the mineral classifier.
- **Debugging** — diagnosing and fixing errors across the codebase.
- **Website scaffolding** — putting together the structure of the
  Streamlit dashboard.

## Citations

**Dataset (MUMDMC2025)** — the granite thin-section photomicrographs used to
train the classifier:

> Amer, B. G., Mousa, H. M., Dawoud, M., & Youssef, A. (2025). A Photomicrographic
> Dataset of Rocks for the Accurate Classification of Minerals. *Scientific Data*,
> 12, 1775. https://doi.org/10.1038/s41597-025-05879-9

**Segment Anything Model (SAM)** — grain segmentation:

> Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T.,
> Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment
> Anything. *arXiv preprint* arXiv:2304.02643. https://arxiv.org/abs/2304.02643

**ResNet-18** — the classifier backbone:

> He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image
> Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern
> Recognition (CVPR)*, 770–778. https://arxiv.org/abs/1512.03385