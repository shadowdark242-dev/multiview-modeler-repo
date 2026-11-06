"""depth_fusion.py - method C2-C: mesh from 8 depth maps + 8 normal maps on a voxel grid, sub-voxel surface.

fuse(views, H, R, N=256, smooth=True, auto_offset=True, log=print) -> V, F, stats
  views: {angle_deg: (depth, normal_cam, mask)} in the aligned frame of live_worker (feet at MARGIN, height = (1-2*MARGIN) R)
    depth      R x R float, world units TOWARD the camera (bigger = closer), nan outside the figure
    normal_cam R x R x 3, camera space (x right, y up, z toward the camera)
    mask       R x R bool
Steps (see the experiment notes):
  1. three states per voxel and view: FREE (in front of the surface or outside the mask), SURFACE (within 3 voxels:
     point-to-plane distance s * n_z), UNKNOWN (behind); depth sampled bilinearly at the voxel's exact projection;
  2. robust fusion: observations further than 1 voxel from the median of the views dropped, weight n_z;
     FREE + observed -> outside, but keeps the measured distance (>= +1/4 voxel); only UNKNOWN -> inside;
  3. dual contouring: one vertex per surface cell from the regularised QEF sum w [N.(X-P)]^2 + lam |X-X0|^2
     over the views that see the same surface (point within 1 voxel of the median, normal within 35 deg);
  4. feature-preserving smoothing along the normal (other surface / big step not mixed), at most 0.3 voxel.
auto_scale / auto_offset: depth maps from a generator have an unknown scale and have an unknown constant per view; it is set so that the depth surface
touches the silhouette hull from inside (10th percentile of hull front - depth).
Measured on ideal maps of a test character, N = 256, R = 1024: mean 0.214 % of height, normals 5.1 deg.
"""
import math, time
import numpy as np
from scipy import sparse
from skimage import measure
from par import pmap

MARGIN = 0.045
OFFSET_Q = 10.0         # percentile of (hull front - depth): the depth surface touches the hull from inside


def _bil(img, r, c, R):
    r0 = np.floor(r).astype(np.int32); c0 = np.floor(c).astype(np.int32); fr = (r - r0).astype(np.float32); fc = (c - c0).astype(np.float32)
    r0 = np.clip(r0, 0, R - 2); c0 = np.clip(c0, 0, R - 2)
    a, b, cc, d = img[r0, c0], img[r0, c0 + 1], img[r0 + 1, c0], img[r0 + 1, c0 + 1]
    with np.errstate(invalid="ignore"):
        v = a * (1 - fc) * (1 - fr) + b * fc * (1 - fr) + cc * (1 - fc) * fr + d * fc * fr
    return np.where(np.isfinite(a) & np.isfinite(b) & np.isfinite(cc) & np.isfinite(d), v, np.nan)


def _cams(a):
    t = math.radians(a)
    return np.array([math.cos(t), math.sin(t), 0.0]), np.array([math.sin(t), -math.cos(t), 0.0])


def _grid(H, N):
    vs = H / N
    return vs, np.arange(-0.45 * H, 0.45 * H, vs) + vs / 2, np.arange(0, H + vs, vs) + vs / 2


def hull_fronts(views, H, R, N):
    """per view: depth (toward the camera) of the front of the silhouette hull at every pixel, -inf where none"""
    from scipy import ndimage
    vs, xs, zs = _grid(H, N); S = R * (1 - 2 * MARGIN) / H; X, Y = np.meshgrid(xs, xs, indexing="ij")
    rows = np.clip(np.round(R - 1 - (zs * S + R * MARGIN)).astype(int), 0, R - 1)
    hull = np.ones((len(xs), len(xs), len(zs)), bool)
    for a, (_, _, m) in views.items():
        right, _ = _cams(a); col = np.round((X * right[0] + Y * right[1]) * S + R / 2 - 0.5).astype(int); ok = (col >= 0) & (col < R)
        hull &= m[rows][:, np.clip(col, 0, R - 1)].transpose(1, 2, 0) & ok[..., None]
    I = np.argwhere(hull); P = np.c_[xs[I[:, 0]], xs[I[:, 1]], zs[I[:, 2]]]; out = {}
    k = 2 * int(math.ceil(S * vs)) + 1                                          # voxel footprint in pixels: fill the gaps
    for a in views:
        right, cam = _cams(a)
        c = np.round(P @ right * S + R / 2 - 0.5).astype(int); r = np.round(R - 1 - (P[:, 2] * S + R * MARGIN)).astype(int)
        ok = (c >= 0) & (c < R) & (r >= 0) & (r < R); fb = np.full((R, R), -np.inf); np.maximum.at(fb, (r[ok], c[ok]), (P @ cam)[ok] + vs / 2)
        out[a] = ndimage.maximum_filter(fb, size=k)
    return out


def auto_offsets(views, H, R, N, fronts=None):
    """constant depth offset per view: the depth surface must touch the silhouette hull from inside"""
    fronts = fronts or hull_fronts(views, H, R, N); off = {}
    for a, (d, _, m) in views.items():
        fb = fronts[a]; sel = m & np.isfinite(d) & np.isfinite(fb)
        off[a] = float(np.percentile((fb - d)[sel], OFFSET_Q)) if sel.sum() > 100 else 0.0
    return off


def auto_scales(views, H, R):
    """unknown depth scale per view (relative depth from a generator): the depth gradient must match the slope the
    normals imply; pixels at depth jumps (overlaps: strand over face, arm over body) are left out - they biased the
    plain estimate by ~1.6x. Tested on degraded maps: 1.6 % mean scale error."""
    S = R * (1 - 2 * MARGIN) / H; out = {}; K = (H / 55) * (1024 / R)   # thresholds tested at H = 55, R = 1024
    from scipy import ndimage
    for a, (d, n, m) in views.items():
        dd = np.where(m & np.isfinite(d), d, 0.0); nz = np.clip(n[..., 2], 1e-3, None)
        gc = -n[..., 0] / nz / S; gr = n[..., 1] / nz / S; dc = np.gradient(dd, axis=1); dr = np.gradient(dd, axis=0)
        jumps = ndimage.binary_dilation((np.abs(dc) > 0.15 * K) | (np.abs(dr) > 0.15 * K), iterations=2)
        ok = ndimage.binary_erosion(m & np.isfinite(d), iterations=3) & (n[..., 2] > 0.4) & (np.abs(gc) < 0.2 * K) & (np.abs(gr) < 0.2 * K) & ~jumps
        den = (gc ** 2 + gr ** 2)[ok].sum()
        out[a] = float((dc * gc + dr * gr)[ok].sum() / den) if ok.sum() > 500 and den > 0 else 1.0
        if not (0.2 < out[a] < 5):
            out[a] = 1.0
    return out


def fuse(views, H, R, N=256, smooth=True, auto_offset=True, auto_scale=False, max_carve=0.0, log=print):
    st = {}; t0 = time.time(); views = dict(views)
    vs, xs, zs = _grid(H, N); S = R * (1 - 2 * MARGIN) / H; TAU = 3 * vs; shape = (len(xs), len(xs), len(zs))
    if auto_scale:
        sc = auto_scales(views, H, R); st["depth_scales"] = {str(int(a)): round(v, 4) for a, v in sc.items()}
        views = {a: (d / sc[a], n, m) for a, (d, n, m) in views.items()}
    fronts = hull_fronts(views, H, R, N)
    if auto_offset:
        off = auto_offsets(views, H, R, N, fronts); st["depth_offsets"] = {str(int(a)): round(o, 4) for a, o in off.items()}
    else:
        off = {a: 0.0 for a in views}
    if True:                                           # depth never in front of the hull (protruding parts are pressed onto it, not carved);
        lim = max_carve * H if max_carve > 0 else np.inf; clipped = {}   # generated depth: not deeper than max_carve * H under it
        for a, (d, n, m) in list(views.items()):
            fb = fronts[a]; dd = d + off[a]; okf = np.isfinite(fb) & np.isfinite(dd)
            nd = np.where(okf, np.clip(dd, fb - lim, fb), dd); clipped[str(int(a))] = round(100 * float((np.abs(nd - dd) > 1e-6)[okf].mean()), 1)
            views[a] = (nd - off[a], n, m)
        st["depth_clipped_pct"] = clipped
    maps = {}
    for a, (d, n, m) in views.items():
        right, cam = _cams(a)
        nw = n[..., :1] * right + n[..., 1:2] * np.array([0, 0, 1.0]) + n[..., 2:3] * cam      # camera -> world normal
        nw /= np.linalg.norm(nw, axis=-1, keepdims=True) + 1e-12
        zb = np.where(m & np.isfinite(d), d + off[a], -np.inf).astype(np.float32)
        maps[a] = (zb, nw.astype(np.float32), right, cam)
    # 1-2. three states + robust point-to-plane fusion
    log("STATUS depth fusion: voxel states 10 %")
    X, Y = np.meshgrid(xs, xs, indexing="ij"); rowf_all = (R - 1 - (zs * S + R * MARGIN)).astype(np.float32)
    F = np.empty(shape, np.float32); nfree = nobs = nunk = 0
    nzimgs = {a: (nb[..., 0] * cam[0] + nb[..., 1] * cam[1]).astype(np.float32) for a, (zb, nb, right, cam) in maps.items()}
    cols = {a: ((X * right[0] + Y * right[1]) * S + R / 2 - 0.5).astype(np.float32) for a, (zb, nb, right, cam) in maps.items()}
    dvs = {a: (X * cam[0] + Y * cam[1])[..., None].astype(np.float32) for a, (zb, nb, right, cam) in maps.items()}
    from par import N as _NP
    CH = max(4, int(2.5e7 * 8 / max(8, _NP) // (shape[0] * shape[1])))   # z slices per chunk: memory ~1-2 GB at any N and core count
    for z0 in range(0, shape[2], CH):
        z1 = min(shape[2], z0 + CH); rowf = rowf_all[z0:z1]; shp = (shape[0], shape[1], z1 - z0)
        def one(item):
            a, (zb, nb, right, cam) = item
            R3 = np.broadcast_to(rowf[None, None], shp); C3 = np.broadcast_to(cols[a][..., None], shp)
            inside = (C3 >= 0) & (C3 <= R - 1) & (R3 >= 0) & (R3 <= R - 1)
            D = _bil(zb, R3, C3, R); nz = np.clip(_bil(nzimgs[a], R3, C3, R), 0.05, 1)
            ri = np.clip(np.round(R3).astype(np.int32), 0, R - 1); ci = np.clip(np.round(C3).astype(np.int32), 0, R - 1)
            empty = ~np.isfinite(zb[ri, ci]) | ~inside
            s = dvs[a] - D
            with np.errstate(invalid="ignore"):
                fr = (empty | (s > TAU)).astype(np.uint8); o = np.isfinite(s) & (np.abs(s) <= TAU) & ~empty
            return fr, np.where(o, s * nz, np.nan).astype(np.float32), np.where(o, nz, 0).astype(np.float32)
        res = pmap(one, maps.items())
        free = np.sum([r_[0] for r_ in res], axis=0, dtype=np.uint8); Bs = np.stack([r_[1] for r_ in res]); Ws = np.stack([r_[2] for r_ in res]); del res
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); med = np.nanmedian(Bs, axis=0)
            keep = np.isfinite(Bs) & (np.abs(Bs - med[None]) <= vs); w = np.where(keep, Ws, 0)
            Sum = np.nansum(np.where(keep, Bs, 0) * w, 0); Wt = w.sum(0); obs = Wt > 0; Fo = Sum / np.maximum(Wt, 1e-9)
        del Bs, Ws, keep, w, med
        F[:, :, z0:z1] = np.where(obs, np.where(free > 0, np.maximum(Fo, 0.25 * vs), Fo), np.where(free > 0, TAU, -TAU))
        nfree += int((free > 0).sum()); nobs += int(obs.sum()); nunk += int(((free == 0) & ~obs).sum())
        del free, obs, Fo, Sum, Wt
        log(f"STATUS depth fusion: voxel states {10 + int(70 * z1 / shape[2])} %")
    del cols, dvs, nzimgs
    st["voxels_free"] = nfree; st["voxels_observed"] = nobs
    st["voxels_unknown_closed"] = nunk; st["field_s"] = round(time.time() - t0, 1)
    # 3. dual contouring with the regularised QEF
    log("STATUS depth fusion: sub-voxel surface 82 %"); t = time.time()
    ins = F < 0
    c8 = [ins[i:ins.shape[0] - 1 + i, j:ins.shape[1] - 1 + j, k:ins.shape[2] - 1 + k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    cnt = np.zeros(c8[0].shape, np.uint8)
    for c_ in c8:
        cnt += c_
    cell = (cnt > 0) & (cnt < 8); del c8, cnt
    I = np.argwhere(cell); nc = len(I); idx = -np.ones(cell.shape, np.int32); idx[tuple(I.T)] = np.arange(nc)
    acc = np.zeros((nc, 3)); num = np.zeros(nc)
    for ax, dvec in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        for oi in (0, 1):
            for oj in (0, 1):
                offs = [(0, oi, oj), (oi, 0, oj), (oi, oj, 0)][ax]
                a0 = I + np.array(offs); a1 = a0 + np.array(dvec)
                f0 = F[tuple(a0.T)]; f1 = F[tuple(a1.T)]; sc = (f0 < 0) != (f1 < 0)
                tt = np.clip(f0 / np.where(f0 - f1 == 0, 1e-9, f0 - f1), 0, 1)
                acc += np.where(sc[:, None], a0 + tt[:, None] * np.array(dvec), 0); num += sc
    X0i = acc / np.maximum(num, 1)[:, None]; X0 = np.c_[xs[0] + X0i[:, 0] * vs, xs[0] + X0i[:, 1] * vs, zs[0] + X0i[:, 2] * vs]
    Ps, Ns, Wv = [], [], []
    for a, (zb, nb, right, cam) in maps.items():
        colf = X0 @ right * S + R / 2 - 0.5; rowf2 = R - 1 - (X0[:, 2] * S + R * MARGIN)
        D = _bil(zb, rowf2, colf, R); s = X0 @ cam - D
        n = np.stack([_bil(nb[..., k], rowf2, colf, R) for k in range(3)], 1); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        with np.errstate(invalid="ignore"):
            nz = n @ cam; ok = np.isfinite(s) & (np.abs(s) <= 1.5 * vs) & (nz > 0.05)
        Ps.append(np.where(ok[:, None], X0 - s[:, None] * cam[None], np.nan)); Ns.append(np.where(ok[:, None], n, 0)); Wv.append(np.where(ok, nz, 0))
    Ps = np.stack(Ps); Ns = np.nan_to_num(np.stack(Ns)); Wv = np.stack(Wv)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); medP = np.nanmedian(Ps, axis=0)
    medN = np.nansum(Ns * Wv[..., None], 0); medN /= np.linalg.norm(medN, axis=1, keepdims=True) + 1e-9
    good = np.isfinite(Ps[..., 0]) & (np.linalg.norm(np.nan_to_num(Ps - medP[None]), axis=-1) <= vs) & ((Ns * medN[None]).sum(-1) > math.cos(math.radians(35)))
    Wv = np.where(good, Wv, 0); Ps = np.nan_to_num(Ps)
    A = np.einsum('vc,vci,vcj->cij', Wv, Ns, Ns); b = np.einsum('vc,vci,vcj,vcj->ci', Wv, Ns, Ns, Ps)
    lam = 0.05 * Wv.sum(0) + 1e-3; A += lam[:, None, None] * np.eye(3); b += lam[:, None] * X0
    Xs = np.linalg.solve(A, b[..., None])[..., 0]
    lo = np.c_[xs[0] + I[:, 0] * vs, xs[0] + I[:, 1] * vs, zs[0] + I[:, 2] * vs] - 0.25 * vs
    Xs = np.clip(Xs, lo, lo + 1.5 * vs)
    st["cells"] = nc; st["cells_without_obs_pct"] = round(100 * float((Wv.sum(0) == 0).mean()), 1)
    quads = []
    for ax in range(3):
        sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3; sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None)
        e = ins[tuple(sl0)] != ins[tuple(sl1)]; E = np.argwhere(e); o1, o2 = [d for d in range(3) if d != ax]; cs = []
        for d1, d2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
            c = E.copy(); c[:, o1] -= 1 - d1; c[:, o2] -= 1 - d2; cs.append(c)
        ok = np.all([(c >= 0).all(1) & (c[:, 0] < cell.shape[0]) & (c[:, 1] < cell.shape[1]) & (c[:, 2] < cell.shape[2]) for c in cs], 0)
        q = np.stack([idx[tuple(np.clip(c, 0, np.array(cell.shape) - 1).T)] for c in cs], 1); ok &= (q >= 0).all(1); quads.append(q[ok])
    Q = np.concatenate(quads); Fq = np.r_[Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]
    gx, gy, gz = np.gradient(F); cen = Xs[Fq].mean(1)                       # every face outward (F > 0 outside)
    gi = np.clip(np.round(np.c_[(cen[:, 0] - xs[0]) / vs, (cen[:, 1] - xs[0]) / vs, (cen[:, 2] - zs[0]) / vs]).astype(int), 0, np.array(F.shape) - 1)
    g = np.c_[gx[tuple(gi.T)], gy[tuple(gi.T)], gz[tuple(gi.T)]]; fnq = np.cross(Xs[Fq[:, 1]] - Xs[Fq[:, 0]], Xs[Fq[:, 2]] - Xs[Fq[:, 0]])
    bad = (fnq * g).sum(1) < 0; Fq[bad] = Fq[bad][:, ::-1]; st["surface_s"] = round(time.time() - t, 1)
    V = Xs
    # 4. feature-preserving smoothing
    if smooth:
        log("STATUS depth fusion: smoothing 95 %"); t = time.time(); n_ = len(V)
        I2 = np.r_[Fq[:, 0], Fq[:, 1], Fq[:, 2]]; J2 = np.r_[Fq[:, 1], Fq[:, 2], Fq[:, 0]]
        Adj = sparse.csr_matrix((np.ones(2 * len(I2), np.float32), (np.r_[I2, J2], np.r_[J2, I2])), shape=(n_, n_)); Adj.data[:] = 1
        rows, cols = Adj.nonzero(); Nobs = np.where(np.isfinite(medN).all(1)[:, None], medN, 0); base = V.copy()
        for _ in range(3):
            fn = np.cross(V[Fq[:, 1]] - V[Fq[:, 0]], V[Fq[:, 2]] - V[Fq[:, 0]]); vn = np.zeros_like(V)
            for k in range(3):
                np.add.at(vn, Fq[:, k], fn)
            vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
            nrm = np.where((np.linalg.norm(Nobs, axis=1) > 0.5)[:, None], Nobs, vn)
            h = ((V[cols] - V[rows]) * nrm[rows]).sum(1); sim = (nrm[rows] * nrm[cols]).sum(1)
            w = np.exp(-((1 - sim) / 0.05) ** 2) * np.exp(-(h / (0.5 * vs)) ** 2)
            V = V + 0.5 * (np.bincount(rows, w * h, n_) / (np.bincount(rows, w, n_) + 1e-9))[:, None] * nrm
            d = V - base; dl = np.linalg.norm(d, axis=1, keepdims=True); V = base + d * np.minimum(1, 0.3 * vs / (dl + 1e-12))
        st["smooth_s"] = round(time.time() - t, 1)
    st["triangles"] = int(len(Fq)); st["total_s"] = round(time.time() - t0, 1)
    return V.astype(np.float32), Fq.astype(np.int32), st
