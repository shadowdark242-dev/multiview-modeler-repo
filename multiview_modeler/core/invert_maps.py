"""invert_maps.py - turn the relief inside out in the given maps (raised <-> recessed), in place.

  python invert_maps.py FILE [FILE ...]

normal_*.png  x and y of the normal flipped (R -> 255 - R, G -> 255 - G) inside the figure; z (B) and alpha kept
depth_*.png   16-bit: v -> 65535 - v inside the figure (depth mirrored about the vertical axis), 0 = background kept
depth_*.npy   d -> -d (NaN background kept)
"""
import sys
import numpy as np
from PIL import Image


def main(files):
    done = 0
    for p in files:
        low = p.lower()
        if low.endswith(".npy"):
            d = np.load(p); np.save(p, -d); done += 1
        elif low.endswith(".png"):
            im = Image.open(p); a = np.asarray(im)
            if a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1):                         # 16-bit depth
                a = a.astype(np.int64).reshape(a.shape[:2]); m = a > 0
                out = np.where(m, np.clip(65535 - a, 1, 65535), 0).astype(np.uint16)
                Image.fromarray(out).save(p)
            else:                                                                          # normal map
                a = a.copy(); m = a[..., 3] > 127 if a.shape[2] == 4 else np.ones(a.shape[:2], bool)
                for k in (0, 1):
                    a[..., k] = np.where(m, 255 - a[..., k], a[..., k])
                Image.fromarray(a).save(p)
            done += 1
        print(f"PROGRESS {done} {len(files)} inverted {p}", flush=True)
    print(f"INVERTED {done}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
