"""bake_maps.py - bake depth + normal maps from ordinary colour views (the 3D-render views made with the 8-view skill).
Runs in the neural venv (PyTorch, CUDA if available). Also usable on its own from a command line.

  python bake_maps.py VIEWS_DIR OUT_DIR [--normals marigold|depth] [--quality fast|normal|best]
                      [--depth-model depth-anything/Depth-Anything-V2-Large-hf] [--normal-model prs-eth/marigold-normals-v1-1]

VIEWS_DIR: view_000.png ... view_315.png (any subset; flat background, e.g. grey #808080, or transparent).
OUT_DIR:   normal_XXX.png  RGBA, camera-space OpenGL normals (red = right, green = up, blue = toward the camera), alpha = silhouette
           depth_XXX.png   16-bit, depth toward the camera in CHARACTER HEIGHTS: value / 65535 - 0.5, 0 = background
                           (plugin: PNG depth range = 1, Bright = far off; Auto depth scale + offset refine it)
           maps_sheet.png  colour | normals | depth of every view, to check by eye
           bake.json       per view: axis flips applied, depth scale fit, share of pixels used
The generators' own conventions are not trusted: normal axes are checked against the silhouette rim (at the left edge the
normal must point left, at the top - up) and the depth sign / scale against the slope the normals imply.
"""
import argparse, json, os, re, sys, time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normals_volume import load_view                                    # the same silhouette the plugin uses

QUALITY = {"fast": (1, 1), "normal": (4, 3), "best": (10, 5)}           # Marigold steps, ensemble size


def say(*a):
    print(*a, flush=True)


def find_views(d):
    out = {}
    for f in sorted(os.listdir(d)):
        mt = re.fullmatch(r"view_(\d{1,3})\.(png|jpg|jpeg|webp)", f, re.I)
        if mt and int(mt.group(1)) < 360:
            out[int(mt.group(1))] = os.path.join(d, f)
    return out


def on_grey(path):
    """RGB on grey #808080 (the generators see what the skill made) + the plugin's silhouette"""
    a = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
    al = a[..., 3:4] / 255.0; rgb = a[..., :3] * al + 128 * (1 - al)
    _, m = load_view(path)
    return Image.fromarray(rgb.round().astype(np.uint8)), m


def delight(img, m):
    """even out drawn lighting before the generators see the picture: broad light gradients (lit side vs shadow side)
    are divided out, thin highlight / shadow lines along edges are compressed; colours (hue, saturation) are kept"""
    a = np.asarray(img).astype(np.float32) / 255.0
    Y = a @ np.array([0.299, 0.587, 0.114], np.float32) + 1e-3
    ys = np.nonzero(m)[0]; hpx = float(ys.max() - ys.min() + 1); sig = max(4.0, 0.03 * hpx)
    w = m.astype(np.float32); Yb = ndimage.gaussian_filter(Y * w, sig) / np.maximum(ndimage.gaussian_filter(w, sig), 1e-3)
    Y1 = Y / np.maximum(Yb, 0.05) * float(np.median(Y[m]))                       # broad light gradients out
    med = ndimage.median_filter(Y1, size=max(3, int(round(0.01 * hpx)) | 1))
    Y2 = med + np.clip(Y1 - med, -0.06, 0.06)                                      # thin highlight / shadow lines compressed
    out = a * (np.clip(Y2, 0.02, 1.5) / Y)[..., None]
    out = np.where(m[..., None], np.clip(out, 0, 1), 128 / 255.0)
    return Image.fromarray((out * 255).round().astype(np.uint8))


def clayify(img, m, k=8):
    """colour view -> white clay: the albedo of every material (chroma clusters, connected regions, their brightest pixels)
    is divided out, what is left is the shading; painted lines and small printed spots are filled from around them"""
    a = np.asarray(img).astype(np.float32) / 255.0
    lin = a ** 2.2; Y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32) + 1e-4
    chrom = lin / (lin.sum(-1, keepdims=True) + 1e-4)
    feat = np.c_[chrom[m][:, :2] * 4.0, np.log(Y[m])[:, None] * 0.15]
    rng = np.random.default_rng(0); smp = feat[rng.choice(len(feat), min(len(feat), 30000), replace=False)]
    c = smp[rng.choice(len(smp), k, replace=False)]
    for _ in range(15):
        lab = ((smp[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([smp[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    L = np.full(m.shape, -1); L[m] = ((feat[:, None] - c[None]) ** 2).sum(-1).argmin(1)
    L = ndimage.median_filter(L, 5)
    ys = np.nonzero(m)[0]; hpx = float(ys.max() - ys.min() + 1); min_area = (0.02 * hpx) ** 2
    alb = np.zeros_like(Y)
    for j in range(k):
        reg, n = ndimage.label((L == j) & m)
        if not n:
            continue
        sizes = ndimage.sum(np.ones_like(Y), reg, np.arange(1, n + 1))
        for r in np.nonzero(sizes >= min_area)[0] + 1:
            sel = reg == r; alb[sel] = np.percentile(Y[sel], 97)
    hole = m & (alb == 0)                                                  # small printed spots: albedo of the neighbours
    if hole.any():
        ii = ndimage.distance_transform_edt(hole | ~m, return_distances=False, return_indices=True); alb[hole] = alb[ii[0][hole], ii[1][hole]]
    sh = np.clip(Y / np.maximum(alb, 1e-3), 0, 1.2)
    w = max(3, int(round(0.005 * hpx)) | 1)
    sh = np.where(hole, ndimage.median_filter(sh, size=2 * w + 1), sh)   # small printed spots only: line removal also ate real creases
    clay = np.clip(0.85 * sh, 0, 1) ** (1 / 2.2)
    out = np.where(m[..., None], np.dstack([clay] * 3), 128 / 255.0)
    return Image.fromarray((out * 255).round().astype(np.uint8))


# ----------------------------------------------------------------------------------- conventions
def fix_normals(n, m):
    """normalise; flip x / y / z so they agree with the silhouette rim and face the camera"""
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    g = ndimage.gaussian_filter(m.astype(np.float32), 2.0); gr, gc = np.gradient(g)
    rim = m & ~ndimage.binary_erosion(m, iterations=3)
    out_x, out_up = -gc, gr                                              # outward direction: right, up
    w = rim & (np.hypot(gc, gr) > 1e-3); flips = []
    for k, o in ((0, out_x), (1, out_up)):
        if (n[..., k][w] * o[w]).sum() < 0:
            n[..., k] *= -1; flips.append("xy"[k])
    inner = ndimage.binary_erosion(m, iterations=6)
    if np.median(n[..., 2][inner if inner.any() else m]) < 0:
        n[..., 2] *= -1; flips.append("z")
    return n, flips


def fit_depth(d, n, m):
    """relative depth -> depth toward the camera in character heights. The depth gradient must match the slope the normals
    imply (gc = -nx/nz, gr = ny/nz in px per px); a negative fit means the generator's sign is reversed. Depth jumps
    (strand over face, arm over body) are left out after a first estimate - they biased the plain fit by ~1.6x."""
    ys = np.nonzero(m)[0]; hpx = float(ys.max() - ys.min() + 1)
    good = m & np.isfinite(d)
    idx = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True); dd = d[idx[0], idx[1]]
    nz = np.clip(n[..., 2], 1e-3, None); gc = -n[..., 0] / nz; gr = n[..., 1] / nz
    dr, dc = np.gradient(dd)
    base = ndimage.binary_erosion(good, iterations=3) & (n[..., 2] > 0.4) & (np.abs(gc) < 2.5) & (np.abs(gr) < 2.5)
    k = None; used = 0.0
    for it in range(3):
        ok = base.copy()
        if k is not None:
            jump = ndimage.binary_dilation((np.abs(dc / k) > 2.5) | (np.abs(dr / k) > 2.5), iterations=2); ok &= ~jump
        den = (gc ** 2 + gr ** 2)[ok].sum()
        if ok.sum() < 500 or den <= 0:
            break
        k = float((dc * gc + dr * gr)[ok].sum() / den); used = float(ok.sum() / max(1, m.sum()))
    if k is None or abs(k) < 1e-12:                                     # no usable slope: assume bigger = closer, 0.3 H range
        lo, hi = np.percentile(dd[m], [2, 98]); k = (hi - lo) / (0.3 * hpx) if hi > lo else 1.0; used = 0.0
    dh = dd / k / hpx                                                    # heights, toward the camera
    dh = dh - np.median(dh[m])
    return np.where(m, dh, np.nan).astype(np.float32), {"scale_px": round(1.0 / k, 5), "sign": "kept" if k > 0 else "flipped",
                                                         "fit_pixels": round(used, 3)}


def normals_from_depth(dh, m):
    """fallback: normals from the depth gradient (camera space, OpenGL); dh in heights"""
    ys = np.nonzero(m)[0]; hpx = float(ys.max() - ys.min() + 1)
    good = m & np.isfinite(dh); idx = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True)
    dp = ndimage.gaussian_filter(np.nan_to_num(dh)[idx[0], idx[1]] * hpx, 1.0)
    dr, dc = np.gradient(dp); n = np.dstack([-dc, dr, np.ones_like(dp)])
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def save(out, ang, n, dh, m):
    rgb = np.where(m[..., None], (n * 0.5 + 0.5) * 255, 128).clip(0, 255).round().astype(np.uint8)
    Image.fromarray(np.dstack([rgb, (m * 255).astype(np.uint8)]), "RGBA").save(os.path.join(out, f"normal_{ang:03d}.png"))
    v = np.where(m, np.clip(0.5 + np.nan_to_num(dh), 1.0 / 65535, 1.0) * 65535, 0).round().astype(np.uint16)
    Image.fromarray(v).save(os.path.join(out, f"depth_{ang:03d}.png"))


def sheet(out, items):
    T = 192; rows = []
    for ang, img, n, dh, m in items:
        col = img.resize((T, T)); nim = Image.open(os.path.join(out, f"normal_{ang:03d}.png")).convert("RGB").resize((T, T))
        lo, hi = np.nanpercentile(dh, [1, 99]); dv = np.where(m, np.clip((np.nan_to_num(dh) - lo) / (hi - lo + 1e-9), 0, 1) * 225 + 30, 0)
        dim = Image.fromarray(dv.astype(np.uint8)).convert("RGB").resize((T, T))
        r = Image.new("RGB", (3 * T, T)); [r.paste(t, (i * T, 0)) for i, t in enumerate((col, nim, dim))]
        ImageDraw.Draw(r).text((4, 4), f"{ang} deg", fill=(255, 255, 255)); rows.append(r)
    if rows:
        cols = 2; W = 3 * T; sh = Image.new("RGB", (cols * W, T * ((len(rows) + 1) // 2)), (40, 40, 40))
        for i, r in enumerate(rows):
            sh.paste(r, ((i % cols) * W, (i // cols) * T))
        sh.save(os.path.join(out, "maps_sheet.png"))


# ----------------------------------------------------------------------------------- generators
class Generators:
    def __init__(self, a):
        import torch
        self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"; self.a = a
        say(f"device {self.dev}" + (f" ({torch.cuda.get_device_name(0)})" if self.dev == "cuda" else " - no CUDA, this will be slow"))
        from transformers import pipeline
        self.depth = pipeline("depth-estimation", model=a.depth_model, device=0 if self.dev == "cuda" else -1)
        self.normal = None
        if a.normals == "marigold":
            from diffusers import MarigoldNormalsPipeline
            dt = torch.float16 if self.dev == "cuda" else torch.float32
            try:
                p = MarigoldNormalsPipeline.from_pretrained(a.normal_model, variant="fp16", torch_dtype=dt)
            except Exception:
                p = MarigoldNormalsPipeline.from_pretrained(a.normal_model, torch_dtype=dt)
            self.normal = p.to(self.dev); self.normal.set_progress_bar_config(disable=True)

    def run_depth(self, img, H, W):
        torch = self.torch
        with torch.no_grad():
            d = self.depth(img)["predicted_depth"]
        d = d.float()
        while d.dim() < 4:
            d = d[None]
        return torch.nn.functional.interpolate(d, size=(H, W), mode="bicubic", align_corners=False)[0, 0].cpu().numpy()

    def run_normals(self, img, H, W):
        steps, ens = QUALITY[self.a.quality]
        gen = self.torch.Generator(device=self.dev).manual_seed(0)
        out = self.normal(img, num_inference_steps=steps, ensemble_size=ens, processing_resolution=768, output_type="np", generator=gen)
        n = np.asarray(out.prediction, np.float32)
        n = n[0] if n.ndim == 4 else n
        if n.shape[0] == 3 and n.shape[-1] != 3:
            n = np.moveaxis(n, 0, -1)
        if n.shape[:2] != (H, W):
            n = np.dstack([np.asarray(Image.fromarray(n[..., k]).resize((W, H), Image.BILINEAR)) for k in range(3)])
        return n


def bake_one(ang, img, m, d_raw, n_raw, out):
    if n_raw is not None:
        n, flips = fix_normals(n_raw, m)
        dh, info = fit_depth(d_raw, n, m)
    else:                                                                # normals from depth: Depth Anything is bigger = closer
        ys = np.nonzero(m)[0]; hpx = float(ys.max() - ys.min() + 1); dd = np.where(m, d_raw, np.nan)
        lo, hi = np.nanpercentile(dd, [2, 98]); dh = (dd - np.nanmedian(dd)) / (hi - lo + 1e-9) * 0.3
        n, flips = normals_from_depth(dh, m), []
        info = {"scale_px": None, "sign": "kept", "fit_pixels": 0.0, "note": "depth range assumed 0.3 H - Auto depth scale cannot help here"}
    save(out, ang, n, dh, m)
    info["flips"] = flips
    return n, dh, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("views"); ap.add_argument("out")
    ap.add_argument("--normals", choices=("marigold", "depth"), default="marigold")
    ap.add_argument("--quality", choices=tuple(QUALITY), default="normal")
    ap.add_argument("--delight", action="store_true", help="even out the drawn lighting before baking (saved as delit_XXX.png)")
    ap.add_argument("--clay", action="store_true", help="convert every view to white clay before baking: colours and painted details out (saved as clay_XXX.png)")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Large-hf")
    ap.add_argument("--normal-model", default="prs-eth/marigold-normals-v1-1")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    views = find_views(a.views)
    if not views:
        sys.exit(f"no view_XXX.png in {a.views}")
    n_steps = len(views) + 2
    say(f"PROGRESS 0 {n_steps} loading the generators (first run downloads ~1.3 GB depth" + (" + ~2 GB normals)" if a.normals == "marigold" else ")"))
    G = Generators(a); report = {}; items = []
    for i, (ang, p) in enumerate(sorted(views.items())):
        say(f"PROGRESS {i + 1} {n_steps} view {ang}")
        img, m = on_grey(p); H, W = m.shape
        if a.clay:
            img = clayify(img, m); img.save(os.path.join(a.out, f"clay_{ang:03d}.png"))
        elif a.delight:
            img = delight(img, m); img.save(os.path.join(a.out, f"delit_{ang:03d}.png"))
        d_raw = G.run_depth(img, H, W)
        n_raw = G.run_normals(img, H, W) if G.normal is not None else None
        n, dh, info = bake_one(ang, img, m, d_raw, n_raw, a.out); report[ang] = info; items.append((ang, img, n, dh, m))
        say(f"view {ang}: {info}")
    say(f"PROGRESS {n_steps - 1} {n_steps} preview sheet"); sheet(a.out, items)
    with open(os.path.join(a.out, "bake.json"), "w") as f:
        json.dump({"views": report, "normals": a.normals, "quality": a.quality, "delight": bool(a.delight), "clay": bool(a.clay), "seconds": round(time.time() - t0, 1)}, f, indent=1)
    say(f"PROGRESS {n_steps} {n_steps} done")
    say("BAKED " + json.dumps(sorted(views)))


if __name__ == "__main__":
    main()
