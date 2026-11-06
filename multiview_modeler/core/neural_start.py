"""neural_start.py - colour image(s) -> rough full mesh with Hunyuan3D 2.0 shape model (runs in the neural venv, CUDA).

  python neural_start.py --image front.png --out raw.npz --repo HUNYUAN_DIR [--variant mini|std|mv] [--fg-ratio 0.85] [--octree 256]
            [--steps 30] [--image-left left.png --image-back back.png]   (mv = three views)

The background is cut by the image itself (alpha, or a smooth-background flood from the border: flat colour, gradient or soft shadow), so rembg is not needed.
Variants: mini (Hunyuan3D-2mini, lightest), std (Hunyuan3D-2, 6 GB for the shape), mv (Hunyuan3D-2mv, front + left + back).
Output: raw.npz with V (any orientation / scale - align_start fixes it), F, optional C (vertex colours 0..1).
"""
import argparse, os, sys, time
import numpy as np
from PIL import Image
from scipy import ndimage


def say(*a):
    print(*a, flush=True)


def background_mask(rgb, step=6.0, band=2, band_tol=30.0, work=512):
    """foreground mask of an image on a smooth background (flat colour, gradient, vignette, soft floor shadow).
    The background is the smooth region that touches the image border: pixels whose neighbour-to-neighbour colour change
    stays under `step` (after a light blur). Edges of the character stop it, so the white eyes or a cream skirt are kept.
    A thin band at the outline that still matches the nearby background colour is then removed. Analysed at `work` px."""
    h0, w0 = rgb.shape[:2]; sc = min(1.0, work / max(h0, w0))
    im = np.asarray(Image.fromarray(rgb).resize((max(8, round(w0 * sc)), max(8, round(h0 * sc))), Image.LANCZOS)).astype(np.float32) if sc < 1 else rgb.astype(np.float32)
    sm = np.stack([ndimage.gaussian_filter(im[..., c], 1.0) for c in range(3)], -1)
    gy = np.abs(np.diff(sm, axis=0)).sum(-1); gx = np.abs(np.diff(sm, axis=1)).sum(-1); g = np.zeros(sm.shape[:2], np.float32)
    g[:-1] = np.maximum(g[:-1], gy); g[1:] = np.maximum(g[1:], gy); g[:, :-1] = np.maximum(g[:, :-1], gx); g[:, 1:] = np.maximum(g[:, 1:], gx)
    lab, _ = ndimage.label(g < step)
    bg = np.isin(lab, list(set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}))
    for _ in range(3):                                     # eat the outline fringe that is still background-coloured
        if not bg.any() or bg.all():
            break
        d, (iy, ix) = ndimage.distance_transform_edt(~bg, return_indices=True)
        near = (d > 0) & (d <= band) & (np.abs(sm - sm[iy, ix]).sum(-1) < band_tol)
        if not near.any():
            break
        bg |= near
    m = ~bg
    m = ndimage.binary_opening(m, iterations=1); lab, n = ndimage.label(m)
    if n:
        m = ndimage.binary_fill_holes(lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1))
    if sc < 1:
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w0, h0), Image.BILINEAR)) > 127
    return m


def cut_background(path):
    """RGBA image with the background removed: the alpha channel if the file has one, otherwise background_mask()"""
    im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
    m = im[..., 3] > 127 if im[..., 3].min() < 250 else background_mask(im[..., :3].astype(np.uint8))
    out = im.copy(); out[..., 3] = m * 255
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def fit_foreground(img, ratio):
    """crop to the silhouette, pad to a square where the object takes `ratio` of the side"""
    a = np.asarray(img); ys, xs = np.nonzero(a[..., 3] > 0)
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; h, w = crop.shape[:2]; side = int(max(h, w) / ratio)
    out = np.zeros((side, side, 4), np.uint8); oy, ox = (side - h) // 2, (side - w) // 2
    out[oy:oy + h, ox:ox + w] = crop
    return Image.fromarray(out, "RGBA")


MODELS = {"mini": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini"),
          "std": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0"),
          "mv": ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv")}


def prepare(path, ratio):
    return fit_foreground(cut_background(path), ratio)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True); ap.add_argument("--out", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--variant", default="mini", choices=list(MODELS))
    ap.add_argument("--image-left", default=""); ap.add_argument("--image-back", default="")
    ap.add_argument("--fg-ratio", type=float, default=0.85)
    ap.add_argument("--octree", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    import torch
    sys.path.insert(0, a.repo)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"PROGRESS 0 4 device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else " - no CUDA, this will be very slow"))
    if a.variant == "mv":
        if not (a.image_left and a.image_back):
            sys.exit("the multi-view model needs the front, left and back images")
        image = {"front": prepare(a.image, a.fg_ratio), "left": prepare(a.image_left, a.fg_ratio), "back": prepare(a.image_back, a.fg_ratio)}
    else:
        image = prepare(a.image, a.fg_ratio)                       # RGBA with transparent background, the model crops by alpha
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    except Exception as e:
        sys.exit("Hunyuan3D code cannot be imported (" + str(e)[:160] + ") - press Install Hunyuan3D again")
    repo_id, sub = MODELS[a.variant]
    say(f"PROGRESS 1 4 loading {repo_id} (first run downloads the weights, a few GB)")
    try:
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo_id, subfolder=sub, use_safetensors=True, device=dev)
    except Exception as e1:
        say("  safetensors weights not found, trying .ckpt: " + str(e1)[:120])
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo_id, subfolder=sub, use_safetensors=False, device=dev)
    say(f"PROGRESS 2 4 generating ({a.steps} steps, grid {a.octree})")
    gen = torch.Generator(device="cpu").manual_seed(2025)
    with torch.no_grad():
        out = pipe(image=image, num_inference_steps=a.steps, octree_resolution=a.octree, num_chunks=20000, generator=gen)
    mesh = out[0] if isinstance(out, (list, tuple)) else out
    V = np.asarray(mesh.vertices, np.float32); F = np.asarray(mesh.faces, np.int32)
    if len(F) == 0:
        sys.exit("the model returned an empty mesh")
    say("PROGRESS 3 4 saving")
    np.savez(a.out, V=V, F=F)
    say(f"PROGRESS 4 4 {len(F)} triangles, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
