"""neural_start.py - colour image(s) -> rough full mesh with Hunyuan3D 2.0 shape model (runs in the neural venv, CUDA).

  python neural_start.py --image front.png --out raw.npz --repo HUNYUAN_DIR [--variant mini|std|mv] [--fg-ratio 0.85] [--octree 256]
            [--steps 30] [--image-left left.png --image-back back.png]   (mv = three views)

The background is cut with the silhouette of the image itself (alpha, or the flat border colour), so rembg is not needed.
Variants: mini (Hunyuan3D-2mini, lightest), std (Hunyuan3D-2, 6 GB for the shape), mv (Hunyuan3D-2mv, front + left + back).
Output: raw.npz with V (any orientation / scale - align_start fixes it), F, optional C (vertex colours 0..1).
"""
import argparse, os, sys, time
import numpy as np
from PIL import Image
from scipy import ndimage


def say(*a):
    print(*a, flush=True)


def cut_background(path, tol=28):
    im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
    if im[..., 3].min() < 250:
        m = im[..., 3] > 127
    else:
        rgb = im[..., :3]; edge = np.r_[rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]; bg = np.median(edge, 0)
        near = np.abs(rgb - bg).sum(-1) < tol; lab, _ = ndimage.label(near)
        border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
        m = ~np.isin(lab, list(border))
    m = ndimage.binary_opening(m, iterations=1); lab, n = ndimage.label(m)
    if n:
        m = lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)
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
