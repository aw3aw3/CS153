"""Generate a synthetic thin-section-ish test image (overlapping blobs)."""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "synthetic_test.png"


def main() -> None:
    rng = np.random.default_rng(0)
    img = np.full((256, 256, 3), 240, dtype=np.uint8)
    for _ in range(8):
        cx, cy = rng.integers(30, 226, size=2)
        r = int(rng.integers(20, 45))
        color = tuple(int(c) for c in rng.integers(60, 220, size=3))
        cv2.circle(img, (int(cx), int(cy)), r, color, -1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), img)
    print(f"wrote {OUT} {img.shape}")


if __name__ == "__main__":
    main()
