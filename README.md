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

A browser opens at `http://localhost:8501`. Drag in a thin-section image and you
get: an **interactive mineral map you can hover** (each grain shows its predicted
mineral, confidence and uncertainty), an uncertainty map, the modal mineralogy
(area % / count %), a per-grain table, and a downloadable JSON report. That's the
whole tool — no other setup needed (weights auto-download on first run).

**Target rock type: granite.** The default mineral set is the `granite` preset —
8 minerals (quartz, plagioclase, microcline, orthoclase, biotite, muscovite,
hornblende, zircon) each described by its **diagnostic XPL appearance** (twinning,
extinction, interference colors). For CLIP these descriptions *are* the
classifier, so they matter a lot. Grains that aren't one of the 8 (or aren't
minerals at all) surface via the uncertainty layer / non-grain detection rather
than being forced into a class.

The classifier is **pluggable** (all backends share one `classify(crops)`
interface, so the pipeline is unchanged when you switch):
- **`clip`** (default) — local CLIP zero-shot. No training/labels; scores each
  crop against the granite descriptions. Works on any rock type. Recommended.
- **`finetuned`** — a CNN trained on labeled thin-section crops (experimental;
  only 5 granite minerals — see *Training* and the caveats below).
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
checkpoints/        Model weights (gitignored)
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
- `--backend {clip,finetuned,claude}` — classifier (default `clip`)
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
(biotite, hornblende, plagioclase, potassium-feldspar, quartz) under PPL/XPL
across 72 stage-rotation angles. Labels are inferred from each file's path.

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

## Notes & caveats

- **Zero-shot (`clip`) accuracy is rough — even with the granite descriptions.**
  Generic CLIP hasn't seen many thin sections, so confidences stay low (~0.2–0.3)
  and entropy high. The diagnostic XPL descriptions in the `granite` preset are
  the right lever and help at the margin (e.g. large quartz grains), but they
  can't fully overcome zero-shot on this domain. The honest signal is the
  uncertainty map, not the point labels.
- **Unidentifiable grains.** Input is assumed granite (the 8 listed minerals).
  Anything else still gets one of the 8 labels, but typically with high entropy /
  low margin — set `--max-entropy` (or the dashboard cutoff) to push those to
  `uncertain` instead of trusting them.
- **The `finetuned` model is a proof of concept, not yet generalizable.** The
  public MUMDMC2025 *sample* contains only **1–2 distinct crystals per mineral**
  (imaged at many rotations), so high validation accuracy mostly reflects
  rotation-robustness on *seen* crystals — not generalization to new thin
  sections. Use `--split-by group` and the full 14,400-image release
  (~20 crystals/mineral) for an honest cross-crystal evaluation.
- **Domain gap in the pipeline.** The CNN is trained on clean, centered
  single-mineral crops, but the pipeline feeds it SAM region-crops (background,
  grain edges, fractures). It maps these to its 5 classes *confidently even when
  wrong*. Next steps: train on SAM-style crops, add a background/"other" class,
  and calibrate confidence (ties into the planned uncertainty milestone).
- **Area % is the petrographic mode.** By the Delesse principle, area fraction in
  a random section estimates volume fraction — the geologically meaningful number,
  reported alongside grain count %.
- Edit `src/minerals.py` to change the candidate mineral set for clip/claude.
