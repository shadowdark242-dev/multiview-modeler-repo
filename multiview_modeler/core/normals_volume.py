"""Shape underlay from normal maps of several views (the agent's own "image -> 3D" step).

  python normals_volume.py OUT_DIR --height 2.2 \
      --view 0:n_front.png --view 45:n_45.png --view 90:n_side.png --view 135:n_135.png \
      --view 180:n_back.png --view 225:n_225.png --view 270:n_270.png --view 315:n_315.png \
      [--color 0:c_front.png ...  (vertex colours for texturing only)] [--res 800] [--refine 40] [--soft 0.8]

Input: camera-space normal maps made by shading_to_normals.py from 3D views (do NOT use normal maps drawn by an image
generator: drawn views disagree with each other). R = right, G = up, B = toward the viewer; the script also
detects DirectX-style G = down), same pose, scale and framing in every view, plain background or alpha.
Angle = camera azimuth: 0 front, 90 = character's left side (character looks LEFT in the picture), 180 back,
45/135/225/315 between. 8 views recommended (6 = 0,45,90,135,180,270 is the minimum for good diagonals).

Pipeline (tested on maps rendered from a reference model: median distance error 0.33 % of height, normal error
8.3° overall / 6.7° on flat areas at 800 px, closed mesh):
  1. masks + alignment of views (height, horizontal centre on the legs);
  2. per view: weighted integration of the normal map into a depth map (soft cuts at occlusions);
  3. depth parts anchored to the silhouette hull; grazing pixels do not carve;
  4. depth hull: a point is inside if in every view it is not in front of that view's surface (soft minimum);
  5. marching cubes in seamless slabs, curvature-adaptive Taubin smoothing (flat areas smoothed, details kept);
  6. vertex refinement so faces follow the observed normals (detail from the maps, ripple removed);
  7. optional colour projection from colour views -> vertex colours (reference for texturing only).

Outputs:
  underlay.obj      closed mesh, units of --height, faces -Y, feet at z = 0 (with vertex colours if --color)
  check.png         ONE check sheet: the underlay shaded from 0, 45, 90, 180 (one visual pass)
  report.json       per view: coverage, integration parts, silhouette agreement; slices (width/depth/front/back
                    per 5 % height); KNOWN DEFECTS to not copy into the model
The underlay is a SHAPE REFERENCE, not the model and not the truth for part boundaries: see the defects list.
Requires numpy, scipy, scikit-image, Pillow. Memory ~2-3 GB at --res 800.
"""
import argparse
import json
import math
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
from skimage import measure

KNOWN_DEFECTS = [
    "Layers that touch in every view merge into one surface: skirt + frill/petticoat, cape over dress, bangs over forehead, "
    "ribbons on hair. Parts come from the part map of the design, never from the underlay geometry or colours.",
    "Gaps and concavities seen from no view are filled: space between bangs and forehead, ear cups, under the skirt, "
    "between arm and body, inside the hood. Build them open as the design shows.",
    "Thin parts (ribbons, ear shells, cape edge, frill) come out thicker and rounded; use the thickness rules, not the underlay.",
    "T-pose arm tips, fingers, tail tip and other parts visible in few views come out short or blobby.",
    "Small details (bells, buttons, stitches) are softened; embossed relief belongs to the texture, not to geometry.",
    "Faint horizontal banding can remain on thin parts; it is reconstruction noise, not a design feature.",
]


def parse(spec):
    ang, path = spec.split(":", 1)
    return float(ang), path


def load_view(path, tol=28):
    im = Image.open(path)
    a = np.asarray(im.convert("RGBA")).astype(float)
    if a[..., 3].min() < 250:
        m = a[..., 3] > 127
    else:
        rgb = a[..., :3]
        edge = np.r_[rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]
        bg = np.median(edge, 0)
        near = np.abs(rgb - bg).sum(-1) < tol
        lab, _ = ndimage.label(near)
        border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
        m = ~np.isin(lab, list(border))
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n:
        m = lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)
    return a[..., :3], m


def align(rgb, m, RES, foot=0.045, top=0.045):
    """Scale so the silhouette height fills RES*(1-foot-top), feet at RES*foot from the bottom, legs centred."""
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    Hpx = RES * (1 - foot - top)
    s = Hpx / (y1 - y0)
    w = max(1, int(round((x1 - x0) * s)))
    # fill the background with the nearest foreground colour so resampling never mixes normals with background
    idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
    rgb = rgb[idx[0], idx[1]]
    crop = Image.fromarray(rgb[y0:y1, x0:x1].astype(np.uint8)).resize((w, int(round(Hpx))), Image.BILINEAR)
    cm = np.asarray(Image.fromarray((m[y0:y1, x0:x1] * 255).astype(np.uint8)).resize((w, int(round(Hpx))), Image.BILINEAR)) > 127
    legs = cm[-max(2, int(cm.shape[0] * 0.12)):]
    cx = np.nonzero(legs)[1].mean() if legs.any() else w / 2
    out = np.zeros((RES, RES, 3)); om = np.zeros((RES, RES), bool)
    oy = int(round(RES * top)); ox = int(round(RES / 2 - cx))
    ys_, xs_ = slice(oy, oy + cm.shape[0]), slice(max(0, ox), min(RES, ox + w))
    src = slice(max(0, -ox), max(0, -ox) + (xs_.stop - xs_.start))
    out[ys_, xs_] = np.asarray(crop).astype(float)[:, src]
    om[ys_, xs_] = cm[:, src]
    return out, om


def decode_normals(rgb, m):
    n = rgb / 255.0 * 2 - 1
    best = None
    for gy in (1, -1):                                    # OpenGL (G up) vs DirectX (G down)
        nn = n * np.array([1, gy, 1.0])
        nz = np.clip(nn[..., 2], 0.15, None)
        p, q = -nn[..., 0] / nz, nn[..., 1] / nz
        ok = m & (nn[..., 2] > 0.15)
        curl = np.abs(np.gradient(p, axis=0) - np.gradient(q, axis=1))[ok].mean()
        if best is None or curl < best[0]:
            best = (curl, gy, nn)
    nn = best[2]
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True) + 1e-9
    return nn, ("opengl" if best[1] == 1 else "directx")


def integrate(m, n, RES, brk=0.6, lab=None):
    nz = n[..., 2]; ok = m & (nz > 0.08)
    p = -n[..., 0] / np.clip(nz, 0.08, None); q = n[..., 1] / np.clip(nz, 0.08, None)
    idx = -np.ones(m.shape, int); ys, xs = np.nonzero(ok); idx[ys, xs] = np.arange(len(ys)); N = len(ys)
    rows, cols, vals, b, E, r = [], [], [], [], [], 0
    for dy, dx, g in ((0, 1, p), (1, 0, q)):
        yy, xx = ys + dy, xs + dx; inb = (yy < RES) & (xx < RES)
        a = idx[ys[inb], xs[inb]]; c = idx[yy[inb], xx[inb]]; good = c >= 0
        a, c = a[good], c[good]
        y0, x0, y1, x1 = ys[inb][good], xs[inb][good], yy[inb][good], xx[inb][good]
        gv = 0.5 * (g[y0, x0] + g[y1, x1]); jump = np.linalg.norm(n[y0, x0] - n[y1, x1], axis=-1)
        w = np.exp(-(jump / 0.25) ** 2) * np.clip(np.minimum(nz[y0, x0], nz[y1, x1]) / 0.3, 0.05, 1)
        cut = (jump > brk) | (np.abs(gv) > 6)
        if lab is not None:                                    # parts are integrated separately
            cut |= lab[y0, x0] != lab[y1, x1]
        a, c, gv, w = a[~cut], c[~cut], np.clip(gv[~cut], -6, 6), w[~cut]
        E.append(np.c_[a, c]); k = len(a); rr = np.arange(r, r + k)
        rows += [rr, rr]; cols += [c, a]; vals += [w, -w]; b.append(gv * w); r += k
    E = np.concatenate(E)
    nc, lab = sparse.csgraph.connected_components(sparse.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(N, N)), directed=False)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, N))
    z = spsolve((A.T @ A + sparse.identity(N) * 1e-6).tocsc(), A.T @ np.concatenate(b))
    return ys, xs, z, lab, nc


def refine_offsets(views, RES, step=2, rows=6, rng=8, passes=2):
    """Per-view horizontal shift (px) so that all silhouettes agree on one vertical axis.
    Uses every `rows`-th row and a voxel grid of `step` px; objective = mean IoU of the re-projected hull."""
    angs = list(views); masks = {a: views[a][0][::rows] for a in angs}
    ext = int(max(np.ptp(np.nonzero(m.any(0))[0]) for m in masks.values()) // 2 + rng + 4)
    g = np.arange(-ext, ext + 1, step, dtype=float); X, Y = np.meshgrid(g, g, indexing="xy")
    cols = {a: X * math.cos(math.radians(a)) + Y * math.sin(math.radians(a)) + RES / 2 - 0.5 for a in angs}
    def score(off):
        ins = np.ones((masks[angs[0]].shape[0],) + X.shape, bool)
        for a in angs:
            c = np.round(cols[a] - off[a]).astype(int); ok = (c >= 0) & (c < RES)
            ins &= masks[a][:, np.clip(c, 0, RES - 1)] & ok[None]
        tot = 0
        for a in angs:
            c = np.clip(np.round(cols[a] - off[a]).astype(int), 0, RES - 1)
            rep = np.zeros_like(masks[a]); rr, yy, xx = np.nonzero(ins); rep[rr, c[yy, xx]] = True
            rep = ndimage.binary_dilation(rep, structure=np.ones((1, step + 1), bool))
            tot += (rep & masks[a]).sum() / max(1, (rep | masks[a]).sum())
        return tot / len(angs)
    off = {a: 0 for a in angs}
    best = score(off)
    for _ in range(passes):
        for a in angs[1:]:                                   # first view fixes the frame
            for d in range(-rng, rng + 1):
                if d == off[a]:
                    continue
                trial = dict(off); trial[a] = d; sc = score(trial)
                if sc > best + 1e-4:
                    best, off = sc, trial
    return off


def vnormals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--view", action="append", required=True, help="angle:normal_map.png")
    ap.add_argument("--color", action="append", default=[], help="angle:colour_view.png -> vertex colours for texturing reference")
    ap.add_argument("--height", type=float, default=1.0, help="character height in output units (e.g. heads)")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--aligned-out", default=None, help="also write the aligned maps (multiview_fit.py input) to this folder")
    ap.add_argument("--aligned-res", type=int, default=2048, help="resolution of the aligned maps (detail of the fit)")
    ap.add_argument("--soft", type=float, default=0.8)
    ap.add_argument("--max-carve", type=float, default=0.025, help="max depth below the silhouette hull, fraction of height (0 = no limit)")
    ap.add_argument("--rim", type=float, default=3.0, help="px (at 800) of the silhouette edge that never carves above --carve-thin-above")
    ap.add_argument("--break-jump", type=float, default=0.6, help="normal jump that cuts the integration (occlusion edges)")
    ap.add_argument("--smooth", type=int, default=12, help="Taubin smoothing iterations")
    ap.add_argument("--smooth-thr", type=float, default=0.004, help="curvature threshold: sharper places are not smoothed")
    ap.add_argument("--refine", type=int, default=40)
    ap.add_argument("--thin", type=float, default=9.0, help="parts thinner than ~2x this (px at 800) do not carve")
    ap.add_argument("--carve-thin-above", type=float, default=None,
                    help="height fraction (0..1): above it thin parts DO carve (head, ears, hair buns); below - arms, tail, ribbons keep the silhouette")
    ap.add_argument("--prealigned", action="store_true", help="maps already share scale/axis (renders): skip alignment")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    RES = a.res
    S = RES * (1 - 0.09) / a.height                       # px per output unit
    views, info = {}, {}
    for spec in a.view:
        ang, path = parse(spec)
        rgb, m = load_view(path)
        if not a.prealigned:
            rgb, m = align(rgb, m, RES)
        elif m.shape != (RES, RES):                           # same frame, other size: resample without edge bleed
            idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
            rgb = np.asarray(Image.fromarray(rgb[idx[0], idx[1]].astype(np.uint8)).resize((RES, RES), Image.BILINEAR)).astype(float)
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((RES, RES), Image.BILINEAR)) > 127
        n, conv = decode_normals(rgb, m)
        views[ang] = (m, n)
        info[str(int(ang))] = {"file": os.path.basename(path), "convention": conv}
    # refine horizontal offsets: maximise silhouette agreement of the hull (coordinate descent, coarse grid)
    shifts = refine_offsets(views, RES) if not a.prealigned else {k: 0 for k in views}
    for ang, dx in shifts.items():
        if dx:
            m, n = views[ang]
            views[ang] = (np.roll(m, dx, axis=1), np.roll(n, dx, axis=1))
        info[str(int(ang))]["offset_px"] = int(dx)
    if a.aligned_out:                                         # same frame at a higher resolution for multiview_fit.py
        os.makedirs(a.aligned_out, exist_ok=True); AR = a.aligned_res
        for spec in a.view:
            ang, path = parse(spec); rgb, m = load_view(path)
            if not a.prealigned:
                rgb, m = align(rgb, m, AR)
            elif m.shape != (AR, AR):
                idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
                rgb = np.asarray(Image.fromarray(rgb[idx[0], idx[1]].astype(np.uint8)).resize((AR, AR), Image.BILINEAR)).astype(float)
                m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((AR, AR), Image.BILINEAR)) > 127
            n, _ = decode_normals(rgb, m); dx = int(round(shifts.get(ang, 0) * AR / RES))
            if dx: m, n = np.roll(m, dx, axis=1), np.roll(n, dx, axis=1)
            out = np.dstack([((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8), (m * 255).astype(np.uint8)])
            Image.fromarray(out).save(os.path.join(a.aligned_out, f"n_{int(ang)}.png"))
    xs = np.arange(RES) - RES / 2 + 0.5
    ext = max(np.abs(xs[np.nonzero(m.any(0))[0]]).max() for m, _ in views.values()) + 3
    gx = np.arange(-ext, ext + 1, 1.0); X, Y = np.meshgrid(gx, gx, indexing="xy")
    P = {ang: (X * math.cos(math.radians(ang)) + Y * math.sin(math.radians(ang)) + RES / 2 - 0.5,
               X * math.sin(math.radians(ang)) - Y * math.cos(math.radians(ang))) for ang in views}
    rows_any = np.nonzero(np.any([m.any(1) for m, _ in views.values()], 0))[0]
    r0, r1 = rows_any.min() - 2, rows_any.max() + 3

    def hull_slab(s, e):
        ins = np.ones((e - s,) + X.shape, bool)
        for b, (m, _) in views.items():
            cb = np.round(P[b][0]).astype(int); ok = (cb >= 0) & (cb < RES)
            ins &= m[s:e][:, np.clip(cb, 0, RES - 1)] & ok[None]
        return ins

    def hull_depth(ang):
        D = np.full((RES, RES), -np.inf); ci = np.clip(np.round(P[ang][0]).astype(int), 0, RES - 1); dep = P[ang][1]
        for s in range(r0, r1, 64):
            e = min(r1, s + 64); rr, yy, xx = np.nonzero(hull_slab(s, e))
            np.maximum.at(D, (rr + s, ci[yy, xx]), dep[yy, xx])
        return D

    depth = {}
    for ang, (m, n) in views.items():
        ys, xs_, z, lab, nc = integrate(m, n, RES, a.break_jump)
        Dh = hull_depth(ang); hd = Dh[ys, xs_]; good = np.isfinite(hd); off = np.zeros(nc)
        for k in range(nc):
            sel = (lab == k) & good
            if sel.any():
                off[k] = np.median(hd[sel] - z[sel])
        zc = z + off[lab]
        if a.max_carve > 0:                                   # integration drift must not eat arms, tail, torso
            zc = np.maximum(zc, np.where(good, hd - a.max_carve * a.height * S, -np.inf))
        D = np.full((RES, RES), np.nan); D[ys, xs_] = np.minimum(zc, np.where(good, hd, np.inf))
        # thin parts (arms, tail, ahoge, ribbons) are almost all rim pixels: their depth is unreliable, so they do
        # not carve - the silhouettes of the other views keep their shape
        thick = ndimage.distance_transform_edt(m) > a.thin * RES / 800
        if a.carve_thin_above is not None:                 # head/ears/buns: depth is reliable there and carries the detail
            rows_ = np.arange(RES)[:, None]; inner = ndimage.distance_transform_edt(m) > a.rim * RES / 800   # never the rim
            thick |= (rows_ < RES - 1 - (a.carve_thin_above * a.height * S + RES * 0.045)) & inner
        depth[ang] = (D, m & (n[..., 2] > 0.15) & thick, m)
        info[str(int(ang))].update({"coverage": round(float(np.isfinite(D)[m].mean()), 3), "integration_parts": int(nc)})
        print(f"view {int(ang)}: parts {nc}, {time.time() - t0:.0f}s", flush=True)

    def sample(img, colf):
        c0 = np.floor(colf).astype(int); f = colf - c0
        return img[:, np.clip(c0, 0, RES - 1)] * (1 - f) + img[:, np.clip(c0 + 1, 0, RES - 1)] * f

    verts, faces, nv, BIG, PAD = [], [], 0, 5.0, 4
    for s in range(r0, r1, 96):
        e = min(r1, s + 98); ss, ee = max(0, s - PAD), min(RES, e + PAD)
        fs = []
        for ang, (D, trust, m) in depth.items():
            col, dep = P[ang]; inside = (col >= 0) & (col <= RES - 1)
            Ds = sample(np.nan_to_num(np.where(trust, D, np.nan)[ss:ee], nan=1e9), col)
            Ms = sample(m[ss:ee].astype(float), col) > 0.5
            fs.append(np.where(Ms & inside[None], np.clip(Ds - dep[None], -BIG, BIG), -BIG).astype(np.float32))
        Fs = np.stack(fs); mn = Fs.min(0)
        field = mn - a.soft * np.log(np.exp(-(Fs - mn) / a.soft).sum(0)) if a.soft > 0 else mn
        field = ndimage.gaussian_filter(field, 0.7)[s - ss:s - ss + (e - s)]
        if field.max() <= 0:
            continue
        v, fc, _, _ = measure.marching_cubes(np.pad(field, ((0, 0), (1, 1), (1, 1)), constant_values=-BIG), 0.0)
        v[:, 1:] -= 1; v[:, 0] += s
        cz = v[fc].mean(1)[:, 0]
        fc = fc[(cz >= s + (0 if s == r0 else 1)) & (cz < s + 97)]
        verts.append(v); faces.append(fc + nv); nv += len(v)
    V = np.concatenate(verts); F = np.concatenate(faces)
    _, first, inv = np.unique(np.round(V * 1000).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    V = V[first]; F = inv.reshape(-1)[F]; F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    W = np.c_[gx[0] + V[:, 2], gx[0] + V[:, 1], RES - 1 - V[:, 0] - RES * 0.045] / S
    vn = vnormals(W, F)
    if (vn * (W - W.mean(0))).sum() < 0:
        F = F[:, [0, 2, 1]]
    # curvature-adaptive Taubin smoothing
    n_ = len(W); I = np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]; J = np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
    Adj = sparse.csr_matrix((np.ones(len(I)), (I, J)), shape=(n_, n_)); Adj.data[:] = 1
    deg = np.asarray(Adj.sum(1)).ravel() + 1e-9
    for it in range(a.smooth):
        if it % 3 == 0:
            vn = vnormals(W, F); avg = Adj @ vn / deg[:, None]; avg /= np.linalg.norm(avg, axis=1, keepdims=True) + 1e-12
            w = np.clip(np.exp(-((1 - np.einsum("ij,ij->i", vn, avg)) / a.smooth_thr) ** 2), 0.08, 1)[:, None]
        for lam in (0.55, -0.58):
            W = W + lam * w * (Adj @ W / deg[:, None] - W)
    # visibility per view (z-buffer of the mesh) -> observed normals / colours
    vn = vnormals(W, F)
    colours = {ang: (load_view(p)[0] if a.prealigned else align(*load_view(p), RES)[0]) for ang, p in map(parse, a.color)} if a.color else {}

    def observe():
        acc = np.zeros_like(W); wsum = np.zeros(len(W)); cacc = np.zeros_like(W); csum = np.zeros(len(W))
        for ang, (m, n) in views.items():
            t = math.radians(ang); right = np.array([math.cos(t), math.sin(t), 0]); cam = np.array([math.sin(t), -math.cos(t), 0])
            col = W @ right * S + RES / 2; row = RES - 1 - (W[:, 2] * S + RES * 0.045); dep = W @ cam * S
            ci, ri = np.clip(col.astype(int), 0, RES - 1), np.clip(row.astype(int), 0, RES - 1)
            zb = np.full((RES, RES), -np.inf); np.maximum.at(zb, (ri, ci), dep)
            facing = vn @ cam
            vis = (dep > zb[ri, ci] - 1.5) & m[ri, ci] & (facing > 0.1)
            nw = n[ri, ci]; nworld = nw[:, :1] * right + nw[:, 1:2] * np.array([0, 0, 1.0]) + nw[:, 2:3] * cam
            wt = np.where(vis, np.clip(facing, 0, 1) ** 2, 0)
            acc += nworld * wt[:, None]; wsum += wt
            if ang in colours:
                cacc += colours[ang][ri, ci] * wt[:, None]; csum += wt
        return acc, wsum, cacc, csum
    acc, wsum, cacc, csum = observe()
    target = np.where(wsum[:, None] > 1e-6, acc / np.maximum(wsum, 1e-9)[:, None], vn)
    target /= np.linalg.norm(target, axis=1, keepdims=True) + 1e-12
    nF = len(F); VF = sparse.csr_matrix((np.ones(nF * 3), (F.ravel(), np.repeat(np.arange(nF), 3))), shape=(len(W), nF))
    cnt = np.asarray(VF.sum(1)).ravel() + 1e-9
    ft = target[F].sum(1); ft /= np.linalg.norm(ft, axis=1, keepdims=True) + 1e-12
    for _ in range(2):
        vt = VF @ ft / cnt[:, None]; vt /= np.linalg.norm(vt, axis=1, keepdims=True) + 1e-12
        ft = vt[F].sum(1); ft /= np.linalg.norm(ft, axis=1, keepdims=True) + 1e-12
    nn = (VF @ np.einsum("fi,fj->fij", ft, ft).reshape(nF, 9)).reshape(-1, 3, 3)
    for _ in range(a.refine):
        c = W[F].mean(1)
        W = W + (VF @ (ft * np.einsum("ij,ij->i", ft, c)[:, None]) - np.einsum("vij,vj->vi", nn, W)) / cnt[:, None]
    # vertex colours (texture reference only - parts are NOT split by colour)
    vcol = None
    if colours:
        vcol = np.where(csum[:, None] > 1e-6, cacc / np.maximum(csum, 1e-9)[:, None], 128.0)
    # write obj
    np.savez(os.path.join(a.out, "underlay.npz"), V=W.astype(np.float32), F=F.astype(np.int32))
    with open(os.path.join(a.out, "underlay.obj"), "w") as f:
        f.write("# shape underlay from normals_volume.py - reference only, see report.json known_defects\n")
        if vcol is not None:
            for p_, c_ in zip(W, vcol / 255.0):
                f.write("v %.5f %.5f %.5f %.3f %.3f %.3f\n" % (*p_, *c_))
        else:
            for p_ in W:
                f.write("v %.5f %.5f %.5f\n" % tuple(p_))
        for t in F + 1:
            f.write("f %d %d %d\n" % tuple(t))
    # slices + silhouette agreement
    sl = []
    for fr in np.arange(0.0, 1.0001, 0.05):
        zc = fr * a.height; band = np.abs(W[:, 2] - zc) < a.height * 0.01
        if band.any():
            p = W[band]
            sl.append({"height_frac": round(float(fr), 2), "width": round(float(np.ptp(p[:, 0])), 4), "depth": round(float(np.ptp(p[:, 1])), 4),
                       "front_most_y": round(float(p[:, 1].min()), 4), "back_most_y": round(float(p[:, 1].max()), 4)})
    vn = vnormals(W, F)
    for ang, (m, _) in views.items():
        t = math.radians(ang); right = np.array([math.cos(t), math.sin(t), 0])
        col = np.clip((W @ right * S + RES / 2).astype(int), 0, RES - 1); row = np.clip((RES - 1 - (W[:, 2] * S + RES * 0.045)).astype(int), 0, RES - 1)
        rep = np.zeros_like(m); rep[row, col] = True; rep = ndimage.binary_closing(rep, iterations=2)
        info[str(int(ang))]["silhouette_iou"] = round(float((rep & m).sum() / max(1, (rep | m).sum())), 3)
    bad = [k for k, v in info.items() if v["silhouette_iou"] < 0.85 or v["coverage"] < 0.8]
    json.dump({"views": info, "views_to_check": bad, "slices_from_floor": sl,
               "known_defects": KNOWN_DEFECTS, "units": "same as --height", "triangles": int(len(F))},
              open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # one check sheet (point splat with shading)
    tiles = []
    for ang in (0, 45, 90, 180):
        t = math.radians(ang); right = np.array([math.cos(t), math.sin(t), 0]); cam = np.array([math.sin(t), -math.cos(t), 0])
        L = 0.7 * cam + np.array([0, 0, 0.6]) - 0.35 * right; L /= np.linalg.norm(L)
        R2 = 360; s2 = R2 / (a.height * 1.08)
        px = (W @ right * s2 + R2 / 2).astype(int); py = (R2 - 1 - (W[:, 2] * s2 + R2 * 0.03)).astype(int)
        g = (40 + 205 * np.clip(vn @ L, 0, 1)).astype(np.uint8); img = np.full((R2, R2, 3), 40, np.uint8)
        o = np.argsort(W @ cam); ok = (px >= 0) & (px < R2 - 1) & (py >= 0) & (py < R2 - 1); o = o[ok[o]]
        for dx in (0, 1):
            for dy in (0, 1):
                img[py[o] + dy, px[o] + dx] = np.stack([g[o]] * 3, -1)
        tiles.append(img)
    Image.fromarray(np.concatenate(tiles, 1)).save(os.path.join(a.out, "check.png"))
    print(f"underlay.obj: {len(W)} verts, {len(F)} tris; views to check: {bad or 'none'}; {time.time() - t0:.0f}s")
    print("views:", ", ".join(f"{k}° iou {v['silhouette_iou']} cov {v['coverage']}" for k, v in info.items()))


if __name__ == "__main__":
    main()
