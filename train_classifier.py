"""Fine-tune a CNN mineral classifier on labeled thin-section photomicrographs.

Built for the MUMDMC2025 dataset (Menoufia University; CC-BY 4.0) — 14,400
PPL/XPL photomicrographs of 5 granite minerals (biotite, hornblende,
plagioclase, potassium-feldspar, quartz), imaged across 72 stage-rotation
angles. Labels are inferred from each file's path, so the exact folder layout
doesn't matter as long as the mineral name appears in the path.

    python train_classifier.py --data-root C:\\Users\\liamr\\mumdmc2025\\extracted
    python train_classifier.py --data-root DIR --epochs 6 --polarization xpl

Produces ``checkpoints/mineral_cnn.pt`` consumed by the ``finetuned`` backend.
"""
from __future__ import annotations

import argparse
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.cnn import build_model, eval_transform, train_transform

# Canonical label -> substrings that identify it in a file path (lowercased).
MINERAL_TOKENS: dict[str, tuple[str, ...]] = {
    "biotite": ("biotite",),
    "hornblende": ("hornblende",),
    "plagioclase": ("plagioclase",),
    "potassium feldspar": (
        "potassium-feldspar", "potassium_feldspar", "potassiumfeldspar",
        "potassium feldspar", "k-feldspar", "kfeldspar", "k_feldspar",
    ),
    "quartz": ("quartz",),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _is_xpl(path: Path) -> bool:
    """In MUMDMC2025, crossed-polarized images live under a ``CN`` (crossed
    nicols) folder; everything else is treated as plane/other."""
    segs = [s.lower() for s in re.split(r"[\\/]+", str(path))]
    return any(s == "cn" or "xpl" in s or "crossed" in s for s in segs)


def label_for(path: Path) -> str | None:
    s = str(path).lower()
    for label, tokens in MINERAL_TOKENS.items():
        if any(t in s for t in tokens):
            return label
    return None


def keep_polarization(path: Path, mode: str) -> bool:
    if mode == "both":
        return True
    if mode == "xpl":
        return _is_xpl(path)
    if mode == "ppl":
        return not _is_xpl(path)
    return True


def scan_dataset(
    root: Path, polarization: str, max_per_class: int, seed: int
) -> tuple[list[tuple[Path, int]], list[str]]:
    by_label: dict[str, list[Path]] = {k: [] for k in MINERAL_TOKENS}
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if not keep_polarization(p, polarization):
            continue
        lab = label_for(p)
        if lab is not None:
            by_label[lab].append(p)

    rng = random.Random(seed)
    labels = sorted(k for k, v in by_label.items() if v)
    if not labels:
        raise RuntimeError(
            f"No labeled images found under {root}. Check the path / polarization filter."
        )
    label_to_idx = {l: i for i, l in enumerate(labels)}

    samples: list[tuple[Path, int, str]] = []
    for lab in labels:
        paths = by_label[lab]
        rng.shuffle(paths)
        if max_per_class > 0:
            paths = paths[:max_per_class]
        # Group key = the image's containing folder. In MUMDMC2025 one folder
        # holds all 72 rotation angles of a SINGLE crystal, so grouping by it
        # lets us hold out whole crystals and avoid rotation leakage.
        samples += [(p, label_to_idx[lab], str(p.parent)) for p in paths]
    rng.shuffle(samples)
    return samples, labels


def grouped_split(
    samples: list[tuple[Path, int, str]], val_frac: float, seed: int
) -> tuple[list, list]:
    """Split so every group (crystal) is entirely in train OR val, never both."""
    groups = sorted({g for _, _, g in samples})
    rng = random.Random(seed)
    rng.shuffle(groups)
    target_val = len(samples) * val_frac
    val_groups: set[str] = set()
    n_val = 0
    for g in groups:
        if n_val >= target_val:
            break
        val_groups.add(g)
        n_val += sum(1 for _, _, gg in samples if gg == g)
    train = [s for s in samples if s[2] not in val_groups]
    val = [s for s in samples if s[2] in val_groups]
    return train, val


class MineralCropDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i][0], self.samples[i][1]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


@torch.no_grad()
def evaluate(model, loader, device, n_classes) -> tuple[float, list[list[int]]]:
    model.eval()
    correct = total = 0
    confusion = [[0] * n_classes for _ in range(n_classes)]
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(dim=-1).cpu()
        for t, p in zip(y.tolist(), pred.tolist()):
            confusion[t][p] += 1
            correct += int(t == p)
            total += 1
    return (correct / total if total else 0.0), confusion


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Directory containing the extracted MUMDMC2025 images")
    ap.add_argument("--model", default="resnet18", help="resnet18|resnet34|resnet50")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="Softens targets to curb overconfidence (better-calibrated)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--polarization", choices=["both", "xpl", "ppl"], default="both")
    ap.add_argument("--max-per-class", type=int, default=1200,
                    help="Cap images/class (rotations are redundant). 0 = use all.")
    ap.add_argument("--split-by", choices=["group", "image"], default="group",
                    help="group=hold out whole crystals (honest); image=random (leaky)")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (keep 0 on Windows unless tested)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/mineral_cnn.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  model={args.model}  polarization={args.polarization}")

    samples, labels = scan_dataset(
        args.data_root, args.polarization, args.max_per_class, args.seed
    )
    counts = Counter(labels[s[1]] for s in samples)
    n_groups = len({s[2] for s in samples})
    print(f"[train] {len(samples)} images, {n_groups} crystals, {len(labels)} classes:")
    for l in labels:
        print(f"          {l:<22} {counts[l]}")

    if args.split_by == "group":
        train_samples, val_samples = grouped_split(samples, args.val_frac, args.seed)
        print(f"[train] split-by=group (no rotation leakage)")
    else:
        n_val = int(len(samples) * args.val_frac)
        val_samples, train_samples = samples[:n_val], samples[n_val:]
        print(f"[train] split-by=image (WARNING: rotations leak across split)")
    print(f"[train] {len(train_samples)} train / {len(val_samples)} val")

    train_ds = MineralCropDataset(train_samples, train_transform(args.img_size))
    val_ds = MineralCropDataset(val_samples, eval_transform(args.img_size))
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model = build_model(args.model, len(labels), pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_acc = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)
        val_acc, _ = evaluate(model, val_ld, device, len(labels))
        dt = time.perf_counter() - t0
        print(f"[train] epoch {epoch}/{args.epochs}  loss={train_loss:.3f}  "
              f"val_acc={val_acc:.3f}  ({dt:.0f}s)")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "labels": labels,
                "model_name": args.model,
                "img_size": args.img_size,
                "val_acc": val_acc,
            }, args.out)

    # Final per-class report from the best model's confusion matrix.
    final_acc, confusion = evaluate(model, val_ld, device, len(labels))
    print(f"\n[train] best val_acc={best_acc:.3f}  (final epoch {final_acc:.3f})")
    print("[train] per-class recall (val):")
    for i, l in enumerate(labels):
        tot = sum(confusion[i])
        rec = confusion[i][i] / tot if tot else 0.0
        print(f"          {l:<22} {rec:.3f}  (n={tot})")
    print(f"[train] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
