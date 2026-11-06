"""silhouette_fit.py - push / pull a start mesh so its outline matches the silhouette of every view.

conform(V, F, masks, H, iters=12, step=0.6, spread=25, smooth=2) -> V, stats
  masks: {angle_deg: bool image R x R} in the aligned frame of live_worker (feet at MARGIN, height = (1-2*MARGIN) R).
Per iteration and view:
  * every vertex that projects outside the silhouette is pulled in (to the nearest silhouette pixel);
  * every contour vertex (normal across the view direction, on the edge of the mesh outline) is moved along the
    signed distance of the silhouette, so a contour that is too far in moves out and one too far out moves in.
The displacements of all views are averaged and spread over the mesh (weighted diffusion along edges), so a whole
arm or the back moves, not only the rim. A light Taubin pass keeps the surface smooth.
"""
import math
import numpy as np
from par import pmap
from scipy import ndimage, sparse

MARGIN = 0.045


def vnormals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def adjacency(F, n):
    I = np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]; J = np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
    A = sparse.csr_matrix((np.ones(len(I), np.float32), (I, J)), shape=(n, n)); A.data[:] = 1
    return A, np.asarray(A.sum(1)).ravel() + 1e-9


def project(V, a, H, R):
    S = R * (1 - 2 * MARGIN) / H; t = math.radians(a)
    right = np.array([math.cos(t), math.sin(t), 0.0]); cam = np.array([math.sin(t), -math.cos(t), 0.0])
    return V @ right * S + R / 2 - 0.5, R - 1 - (V[:, 2] * S + R * MARGIN), right, cam, S


def coverage(col, row, F, R):
    """outline of the mesh in one view: vertices, edge midpoints and face centres splatted, then closed"""
    pc = [col, (col[F[:, 0]] + col[F[:, 1]]) / 2, (col[F[:, 1]] + col[F[:, 2]]) / 2, (col[F[:, 2]] + col[F[:, 0]]) / 2,
          col[F].mean(1)]
    pr = [row, (row[F[:, 0]] + row[F[:, 1]]) / 2, (row[F[:, 1]] + row[F[:, 2]]) / 2, (row[F[:, 2]] + row[F[:, 0]]) / 2,
          row[F].mean(1)]
    c = np.clip(np.round(np.concatenate(pc)).astype(int), 0, R - 1); r = np.clip(np.round(np.concatenate(pr)).astype(int), 0, R - 1)
    m = np.zeros((R, R), bool); m[r, c] = True
    return ndimage.binary_closing(ndimage.binary_dilation(m), iterations=1) & ndimage.binary_dilation(m, iterations=2) | m


def signed_distance(m):
    """> 0 outside the silhouette, < 0 inside (pixels)"""
    return ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)


def bilinear(img, c, r):
    R0, R1 = img.shape; c = np.clip(c, 0, R1 - 1.001); r = np.clip(r, 0, R0 - 1.001)
    c0 = np.floor(c).astype(int); r0 = np.floor(r).astype(int); fc = c - c0; fr = r - r0
    return (img[r0, c0] * (1 - fc) * (1 - fr) + img[r0, c0 + 1] * fc * (1 - fr) +
            img[r0 + 1, c0] * (1 - fc) * fr + img[r0 + 1, c0 + 1] * fc * fr)


def iou(masks, V, F, H):
    out = {}
    for a, m in masks.items():
        col, row, *_ = project(V, a, H, m.shape[0]); cv = coverage(col, row, F, m.shape[0])
        out[a] = float((cv & m).sum() / max(1, (cv | m).sum()))
    return out


def conform(V, F, masks, H, iters=12, step=0.6, spread=25, smooth=2, contour_cos=0.35, max_px=6.0, log=None):
    V = np.asarray(V, np.float64).copy(); F = np.asarray(F, np.int64); n = len(V)
    A, deg = adjacency(F, n)
    sd = {a: signed_distance(m) for a, m in masks.items()}
    grad = {a: np.gradient(ndimage.gaussian_filter(s, 1.0)) for a, s in sd.items()}      # (d/drow, d/dcol)
    for it in range(iters):
        vn = vnormals(V, F)

        def one(am):
            a, m = am
            R = m.shape[0]; col, row, right, cam, S = project(V, a, H, R)
            cv = coverage(col, row, F, R)
            edge = ndimage.distance_transform_edt(cv)            # distance to the outline of the mesh (inside it)
            s = bilinear(sd[a], col, row)                         # silhouette signed distance at each vertex
            gr = bilinear(grad[a][0], col, row); gc = bilinear(grad[a][1], col, row)
            gl = np.sqrt(gr ** 2 + gc ** 2) + 1e-9; gr /= gl; gc /= gl
            ci = np.clip(np.round(col).astype(int), 0, R - 1); ri = np.clip(np.round(row).astype(int), 0, R - 1)
            contour = (np.abs(vn @ cam) < contour_cos) & (edge[ri, ci] <= 1.5)
            act = (s > 0.5) | contour                             # outside -> in; contour -> onto the outline
            amt = -np.clip(s, -max_px, max_px)                     # pixels along the outward gradient
            dpx_c = amt * gc; dpx_r = amt * gr
            d3 = (dpx_c[:, None] * right[None] - dpx_r[:, None] * np.array([0, 0, 1.0])[None]) / S
            w = act * (1.0 + contour)
            return d3 * w[:, None], w
        parts = pmap(one, masks.items()); D = sum(p_[0] for p_ in parts); W = sum(p_[1] for p_ in parts)
        D = np.where(W[:, None] > 0, D / np.maximum(W, 1e-9)[:, None], 0.0); Wn = (W > 0).astype(np.float64)
        num, den = D * Wn[:, None], Wn.copy()                     # weighted diffusion: spread the moves over the mesh
        for _ in range(spread):
            num = 0.5 * num + 0.5 * (A @ num) / deg[:, None]; den = 0.5 * den + 0.5 * (A @ den) / deg
        Ds = num / np.maximum(den, 1e-6)[:, None] * np.clip(den / 0.05, 0, 1)[:, None]
        V += step * Ds
        for _ in range(smooth):
            for lam in (0.5, -0.53):
                V += lam * (A @ V / deg[:, None] - V)
        if log:
            sc = iou(masks, V, F, H); log(f"STATUS silhouette fit {it + 1}/{iters}, IoU {np.mean(list(sc.values())):.3f}")
    return V, iou(masks, V, F, H)
