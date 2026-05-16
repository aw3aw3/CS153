# CS153 — Thin Section Grain Segmentation

Computer-vision pipeline that ingests a smartphone photograph of a
petrographic thin section (cross-polarized light) and segments individual
mineral grains. This milestone covers ingest + SAM-based segmentation only;
classification, uncertainty quantification, and the project dashboard come
later.

## Layout

```
src/             Python package (segmentation logic)
data/            Input images (gitignored)
outputs/         Per-run segmentation outputs (gitignored)
checkpoints/     SAM model weights (gitignored, auto-downloaded)
notebooks/       Exploratory notebooks
segment.py       CLI entry point
```

## Setup

Python 3.10+ recommended. Heavy install (~2-3 GB for torch + SAM).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The SAM checkpoint (`sam_vit_b_01ec64.pth`, ~375 MB) is downloaded on first
run into `checkpoints/`.

## Usage

```powershell
python segment.py data\my_thin_section.jpg
```

Outputs land in `outputs/<image_stem>/`:
- `overlay.png` — original image with grain boundaries drawn
- `masks/grain_NNN.png` — one binary mask per grain
- `manifest.json` — per-grain metadata (bbox, area, mask path)

Flags:
- `--model {vit_b,vit_l,vit_h}` — SAM backbone (default `vit_b`)
- `--min-area N` — drop masks smaller than N pixels
- `--out DIR` — override output dir
