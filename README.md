# CS153 — Final Project Implementation

Computer-vision pipeline that ingests a photograph of a petrographic thin
section (cross-polarized light), **segments** individual mineral grains,
**classifies** each grain by mineral, and reports the thin section's
**modal mineralogy** (mineral distribution).

```
image  →  SAM segmentation  →  per-grain crops  →  mineral classifier
       →  count% / area% summary  +  visual outputs
```

The classifier is **pluggable** (all backends share one `classify(crops)`
interface, so the pipeline is unchanged when you switch):
- **`clip`** (default) — local CLIP zero-shot. No training/labels; scores each
  crop against text prompts per mineral. Runs today.
- **`finetuned`** — a CNN trained on labeled thin-section crops
  (see *Training* below). Higher accuracy in-domain.
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
- `--minerals {default,ultramafic}` — candidate mineral vocabulary (clip/claude)
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

- **Zero-shot (`clip`) accuracy is rough.** Generic CLIP hasn't seen many thin
  sections, so labels are approximate and confidences low (~0.2–0.3). Useful as a
  no-data default and for out-of-vocabulary rocks (e.g. ultramafic).
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
