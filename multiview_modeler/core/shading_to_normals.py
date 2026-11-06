"""Normal map FROM a shaded 3D view (measurement, not generation): shape-from-shading with a silhouette prior.

  python shading_to_normals.py VIEW.png OUT_NORMAL.png [--work 360] [--iters 400] [--light auto|x,y,z]

Input: a "3D view" of the character - a matte render (vinyl/clay figure look), flat material colours, ONE soft
key light from the front-top side + ambient, plain background, no outlines, no cast shadows, no strong highlights.
Output: camera-space normal map (R = right, G = up, B = toward viewer; alpha = silhouette), same size as the input,
ready for normals_volume.py, plus OUT_NORMAL.json (estimated light, fit residual, confidence).

How it works (deterministic - it cannot invent parts that are not in the picture):
  1. silhouette from the background; material colours clustered by chroma; albedo per cluster from its brightest
     pixels -> shading = luminance / albedo;
  2. light direction + ambient estimated by least squares against a rounded "inflated" silhouette shape;
  3. a depth map is optimised so that its Lambert shading matches the picture (L-BFGS), regularised by
     curvature smoothness that is relaxed across material borders (where depth may jump);
  4. normals of the optimised depth, upsampled to the input size.
Limits: shading cannot see what is hidden; very dark materials (black hair) carry little shading signal and
lean on the prior; specular metal (bells) is down-weighted. Requires numpy, scipy, Pillow.
"""
import argparse
import json
import math

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.optimize import minimize


def silhouette(a, tol=26):
    edge = np.r_[a[0], a[-1], a[:, 0], a[:, -1]]
    bg = np.median(edge, 0)
    near = np.abs(a - bg).sum(-1) < tol
    lab, _ = ndimage.label(near)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    m = ~np.isin(lab, list(border))
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    return lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1) if n else m


def inflate(m):
    d = ndimage.distance_transform_edt(m)
    R = np.maximum(ndimage.maximum_filter(d, size=max(3, int(d.max()))), 1e-6)
    R = ndimage.gaussian_filter(R, 3)
    return np.sqrt(np.clip(R ** 2 - (R - np.minimum(d, R)) ** 2, 0, None)) * m


def grads(z):
    p = np.zeros_like(z); q = np.zeros_like(z)
    p[:, :-1] = z[:, 1:] - z[:, :-1]; q[:-1] = z[1:] - z[:-1]
    return p, q


def normals_from(z):
    p, q = grads(z)
    N = np.sqrt(1 + p * p + q * q)
    return np.stack([-p / N, q / N, 1 / N], -1)


def solve(I, m, Lm, amb, gain, n0):
    Y = (I[m] - amb) / gain; lit = Y > 0.04
    n = n0[m].copy(); full = lit.all(1)
    n[full] = Y[full] @ np.linalg.pinv(Lm).T
    for k in (0, 1, 2):                       # one channel in shadow: two lights + weak prior
        sel = (~full) & (lit.sum(1) == 2) & (~lit[:, k])
        if sel.any():
            keep = [c for c in range(3) if c != k]
            A = np.r_[Lm[keep], 0.3 * np.eye(3)]
            B = np.c_[Y[sel][:, keep], 0.3 * n0[m][sel]]
            n[sel] = B @ np.linalg.pinv(A).T
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    n[n[:, 2] < 0, 2] *= -1
    return n


def rgb_lights(im, M, prompted=None):
    """Photometric stereo from ONE image lit by three coloured lights (R, G, B channels = three lights).
    Light directions are refined from the picture (rim pixels have known sideways normals + rounded prior)."""
    I = im / 255.0
    m = M
    bgv = np.median(np.r_[I[0], I[-1], I[:, 0], I[:, -1]], 0)
    z0 = inflate(m); n0 = normals_from(ndimage.gaussian_filter(z0, 1.0))
    dist = ndimage.distance_transform_edt(m); gy, gx = np.gradient(ndimage.gaussian_filter(dist, 1.5))
    rim = m & (dist < 2.5) & (np.hypot(gx, gy) > 0.2)
    nr = np.stack([-gx, gy, np.zeros_like(gx)], -1); nr /= np.linalg.norm(nr, axis=-1, keepdims=True) + 1e-9
    nprior = np.where(rim[..., None], nr, n0)
    dirs = prompted if prompted is not None else np.array([[-0.75, 0.15, 0.64], [0.0, 0.8, 0.6], [0.75, 0.15, 0.64]])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    lights, amb, gain = [], [], []
    inner = m & ~ndimage.binary_erosion(m, iterations=1)
    for c in range(3):
        best = None
        base = dirs[c]
        for th in np.radians(np.arange(0, 90, 4)):
            for ph in np.radians(np.arange(0, 360, 10)):
                l = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
                if l @ base < 0.5:                                    # stay near the prompted direction
                    continue
                d = np.clip(nprior @ l, 0, None)[m]; A = np.c_[np.ones(d.size), d]
                coef, *_ = np.linalg.lstsq(A, I[..., c][m], rcond=None); r = ((A @ coef - I[..., c][m]) ** 2).sum()
                if coef[1] > 0 and (best is None or r < best[0]):
                    best = (r, l, coef)
        lights.append(best[1]); amb.append(best[2][0]); gain.append(best[2][1])
    Lm = np.array(lights); amb = np.array(amb); gain = np.array(gain)
    for it in range(4):                       # alternate: normals <-> light intensities / directions
        n_cur = solve(I, m, Lm, amb, gain, n0)
        for c in range(3):
            d = np.clip(n_cur @ Lm[c], 0, None); sel = d > 0.05
            A = np.c_[np.ones(sel.sum()), d[sel]]; coef, *_ = np.linalg.lstsq(A, I[..., c][m][sel], rcond=None)
            if coef[1] > 0.05:
                amb[c], gain[c] = coef
            # refine direction: least squares l = argmin |(I - amb)/gain - N l| on lit pixels
            y = (I[..., c][m] - amb[c]) / gain[c]; lit = y > 0.05
            l = np.linalg.lstsq(n_cur[lit], y[lit], rcond=None)[0]; Lm[c] = l / (np.linalg.norm(l) + 1e-9)
    Y = (I[m] - amb) / gain
    lit = Y > 0.04; full = lit.all(1)
    n = solve(I, m, Lm, amb, gain, n0)
    out = np.zeros(I.shape); out[m] = n
    # light denoise that keeps edges: average with neighbours of similar normal
    sm = np.stack([ndimage.gaussian_filter(out[..., k] * m, 0.8) for k in range(3)], -1)
    wsum = ndimage.gaussian_filter(m.astype(float), 0.8)[..., None] + 1e-9
    out = np.where(m[..., None], sm / wsum, 0); out /= np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9
    t = np.clip(1 - dist / 2.5, 0, 1)[..., None] * 0.5
    out = out * (1 - t) + nr * t * m[..., None]; out /= np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9
    info = {"mode": "rgb-lights", "lights_camera_space": [[round(float(v), 3) for v in l] for l in Lm],
            "ambient": [round(float(v), 3) for v in amb], "gain": [round(float(v), 3) for v in gain],
            "fully_lit_share": round(float(full.mean()), 3)}
    return out * m[..., None], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("view"); ap.add_argument("out")
    ap.add_argument("--work", type=int, default=360, help="working height in px for the optimisation")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--smooth", type=float, default=1.0, help="curvature regulariser weight")
    ap.add_argument("--light", default="auto")
    ap.add_argument("--k", type=int, default=10, help="material colour clusters")
    ap.add_argument("--clay", action="store_true", help="uniform grey clay render: one material, no colour clustering")
    ap.add_argument("--rgb-lights", action="store_true", help="input is a white clay figure lit by red (left), green (top), blue (right) lights")
    a = ap.parse_args()
    im = np.asarray(Image.open(a.view).convert("RGB")).astype(float)
    M = silhouette(im)
    if a.rgb_lights:
        n, info = rgb_lights(im, M)
        out = np.dstack([((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8), (M * 255).astype(np.uint8)])
        Image.fromarray(out).save(a.out); json.dump(info, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
        print("normals ->", a.out, info); return
    ys, xs = np.nonzero(M); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = a.work / (y1 - y0)
    W = max(8, int((x1 - x0) * s)); Hh = a.work
    crop = Image.fromarray(im[y0:y1, x0:x1].astype(np.uint8)).resize((W, Hh), Image.BILINEAR)
    img = np.pad(np.asarray(crop).astype(float), ((4, 4), (4, 4), (0, 0)), mode="edge")
    m = np.pad(np.asarray(Image.fromarray((M[y0:y1, x0:x1] * 255).astype(np.uint8)).resize((W, Hh), Image.BILINEAR)) > 127, 4)
    m = ndimage.binary_erosion(m, iterations=1) | (m & ~ndimage.binary_erosion(m, iterations=1))
    lum = img @ np.array([0.299, 0.587, 0.114]) + 1.0
    # material clusters by chroma (+ a little brightness), albedo = bright percentile per cluster
    chroma = img / (img.sum(-1, keepdims=True) + 3) * 3
    feat = np.concatenate([chroma, np.log(lum)[..., None] * 0.15], -1)
    X = feat[m]; rng = np.random.default_rng(0); cen = X[rng.choice(len(X), a.k, replace=False)]
    for _ in range(15):
        lab = np.argmin(((X[:, None] - cen[None]) ** 2).sum(-1), 1)
        cen = np.array([X[lab == j].mean(0) if np.any(lab == j) else cen[j] for j in range(a.k)])
    L = np.full(m.shape, -1); L[m] = lab
    L = np.where(m, ndimage.median_filter(np.where(m, L, 0), size=3), -1)
    # prior shape first: albedo is estimated against it (luminance ~ albedo * (ambient + max(0, n.l)))
    z0 = inflate(m) * 0.9
    n0 = normals_from(z0)
    wmat = np.zeros(m.shape)
    uniform = a.clay
    if uniform:
        L[m] = 0
    spec = (lum > np.percentile(lum[m], 99.5))
    labels = [j for j in range(a.k) if (L == j).sum() >= 20]
    # shading relative to the brightest part of each material (= surface facing the light): S in [ambient, 1]
    S = np.zeros(m.shape); wmat = np.zeros(m.shape)
    for j in labels:
        sel = L == j
        top = np.percentile(lum[sel & ~spec], 98)
        S[sel] = lum[sel] / max(top, 1.0)
        wmat[sel] = np.clip(top / 70.0, 0.15, 1.0)              # dark materials: weak signal
    S = np.clip(S, 0, 1.2)
    reliable = m & (wmat > 0.5) & ~spec
    amb = float(np.clip(np.percentile(S[reliable] if reliable.any() else S[m], 2), 0.05, 0.8))
    kd = 1.0 - amb
    w = wmat * m * ~spec
    if a.light == "auto":                                         # light direction: best Lambert fit of the prior shape
        best = None
        for th in np.radians(np.arange(0, 80, 5)):
            for ph in np.radians(np.arange(0, 360, 15)):
                l = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
                d = np.clip(n0 @ l, 0, None)[m]
                A = np.c_[np.ones(d.size), d]; coef, *_ = np.linalg.lstsq(A * w[m][:, None], S[m] * w[m], rcond=None)
                r = ((A @ coef - S[m]) ** 2 * w[m]).sum()
                if coef[1] > 0 and (best is None or r < best[0]):
                    best = (r, l)
        light = best[1]
    else:
        light = np.array([float(v) for v in a.light.split(",")]); light /= np.linalg.norm(light)
    lx, ly, lz = light
    # curvature regulariser, relaxed across material borders
    h, wdt = m.shape; idx = -np.ones(m.shape, int); yy, xx = np.nonzero(m); idx[yy, xx] = np.arange(len(yy)); N = len(yy)
    border = ndimage.binary_dilation(np.abs(np.diff(np.pad(L, ((0, 0), (0, 1)), mode="edge"), axis=1)) > 0) | \
             ndimage.binary_dilation(np.abs(np.diff(np.pad(L, ((0, 1), (0, 0)), mode="edge"), axis=0)) > 0)
    rows, cols, vals = [], [], []
    for (dy, dx) in ((0, 1), (1, 0)):
        ok = (yy - dy >= 0) & (yy + dy < h) & (xx - dx >= 0) & (xx + dx < wdt)
        c = idx[yy[ok], xx[ok]]; a1 = idx[yy[ok] - dy, xx[ok] - dx]; a2 = idx[yy[ok] + dy, xx[ok] + dx]
        good = (a1 >= 0) & (a2 >= 0); c, a1, a2 = c[good], a1[good], a2[good]
        wt = np.where(border[yy[ok][good], xx[ok][good]], 0.15, 1.0)
        rows += [c, c, c]; cols += [a1, c, a2]; vals += [wt, -2 * wt, wt]
    off = 0
    R_, C_, V_ = [], [], []
    for k in range(0, len(rows), 3):
        n_ = len(rows[k]); rid = np.arange(off, off + n_)
        for t in range(3):
            R_.append(rid); C_.append(cols[k + t]); V_.append(vals[k + t])
        off += n_
    Lap = sparse.csr_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))), shape=(off, N))
    lam, mu = a.smooth, 0.002
    zinit = z0[m].copy()
    edge_band = m & ~ndimage.binary_erosion(m, iterations=3)
    wv = w.copy(); wv[edge_band] *= 0.2                        # shading at the rim is unreliable
    def energy(zv):
        z = np.zeros(m.shape); z[m] = zv
        p, q = grads(z)
        Nn = np.sqrt(1 + p * p + q * q); u = -p * lx + q * ly + lz; d = u / Nn
        sg = 1 / (1 + np.exp(-12 * d)); f = np.log1p(np.exp(12 * d)) / 12
        r = amb + kd * f - S
        E1 = (wv * r * r).sum()
        g = 2 * wv * r * kd * sg
        dp = g * (-lx / Nn - u * p / Nn ** 3); dq = g * (ly / Nn - u * q / Nn ** 3)
        # p,q valid only where both pixels are inside
        vp = np.zeros_like(m); vp[:, :-1] = m[:, :-1] & m[:, 1:]; vq = np.zeros_like(m); vq[:-1] = m[:-1] & m[1:]
        dp *= vp; dq *= vq
        gz = np.zeros(m.shape)
        gz[:, 1:] += dp[:, :-1]; gz[:, :-1] -= dp[:, :-1]
        gz[1:] += dq[:-1]; gz[:-1] -= dq[:-1]
        Lz = Lap @ zv
        E2 = lam * (Lz ** 2).sum(); g2 = 2 * lam * (Lap.T @ Lz)
        E3 = mu * ((zv - zinit) ** 2).sum(); g3 = 2 * mu * (zv - zinit)
        return E1 + E2 + E3, gz[m] + g2 + g3
    E0 = energy(zinit)[0]
    res = minimize(energy, zinit, jac=True, method="L-BFGS-B", options={"maxiter": a.iters})
    z = np.zeros(m.shape); z[m] = res.x
    n = normals_from(ndimage.gaussian_filter(z, 0.6)) * m[..., None]
    # rim: blend toward the silhouette's sideways normal (physically the surface turns away at the outline)
    dist = ndimage.distance_transform_edt(m); gy, gx = np.gradient(ndimage.gaussian_filter(dist, 1.5))
    out2 = np.stack([-gx, gy, np.zeros_like(gx)], -1); out2 /= np.linalg.norm(out2, axis=-1, keepdims=True) + 1e-9
    t = np.clip(1 - dist / 3.0, 0, 1)[..., None] * 0.6
    n = n * (1 - t) + out2 * t; n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n = n[4:-4, 4:-4]; mm = m[4:-4, 4:-4]
    # back to input size
    full = np.zeros(im.shape); fm = np.zeros(M.shape, bool)
    up = np.asarray(Image.fromarray(((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)).resize((x1 - x0, y1 - y0), Image.BILINEAR)).astype(float)
    full[y0:y1, x0:x1] = up; fm[:] = M
    out = np.dstack([full.astype(np.uint8), (fm * 255).astype(np.uint8)])
    Image.fromarray(out).save(a.out)
    info = {"light_camera_space": [round(float(v), 3) for v in light], "ambient": round(float(amb), 3), "diffuse": round(float(kd), 3),
            "energy_start": round(float(E0), 2), "energy_end": round(float(res.fun), 2), "iterations": int(res.nit),
            "dark_share": round(float((wmat[4:-4, 4:-4][mm] < 0.4).mean()), 3),
            "note": "dark_share = part of the figure with weak shading signal (normals there lean on the prior)"}
    json.dump(info, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print("normals ->", a.out, info)


if __name__ == "__main__":
    main()
