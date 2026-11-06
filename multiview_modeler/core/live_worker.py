"""live_worker.py - keeps maps, integrated depth and hulls in memory; rebuilds the mesh when parameters change.

Live mode:  python live_worker.py --live            (requests: one JSON per line on stdin, the newest one wins)
Final mode: python live_worker.py --final job.json  (full resolution volume + aligned maps for multiview_fit.py)

Request: {"id": 3, "out": "preview.npz", "height": 2.2, "res": 320, "fit_res": 512,
          "views": [{"angle": 0, "path": "...", "enabled": true, "dx": 0, "dy": 0, "fmt": "auto"}, ...],
          "p": {"max_carve": 0.025, "carve_above": 0.56, "use_carve": true, "thin": 9, "rim": 3, "soft": 0.8,
                "break_jump": 0.6, "smooth": 12, "smooth_thr": 0.004,
                "fit_iters": 2, "lam0": 0.3, "lam1": 0.1, "mu": 0.05, "agree": 25, "relax": 0.3}}
Output lines: STATUS <text> | MESH <id> <path> <json stats> | ERR <id> <text>
"""
import json, math, os, queue, sys, threading, time, traceback
import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normals_volume import load_view, align, decode_normals, integrate, refine_offsets, vnormals   # noqa: E402
import mesh_check as mc                                                                       # noqa: E402
import multiview_fit as mvf                                                                   # noqa: E402
from par import pmap                                                                           # noqa: E402

MARGIN = 0.045


def _excess(P, masks, H):
    """how far each point is outside the silhouettes, as a fraction of the height; the worst view counts, 0 = inside all"""
    from silhouette_fit import project
    out = np.zeros(len(P))
    for a, m in masks.items():
        R = m.shape[0]; dist = ndimage.distance_transform_edt(~m)
        col, row, *_ = project(P, a, H, R)
        ci = np.clip(np.round(col).astype(int), 0, R - 1); ri = np.clip(np.round(row).astype(int), 0, R - 1)
        over = np.hypot(col - ci, row - ri)                        # how far beyond the image frame the point is
        out = np.maximum(out, (dist[ri, ci] + over) / (R * (1 - 2 * MARGIN)))
    return out


def _keep_used(V, F, keep_tri):
    Fk = F[keep_tri]; used = np.unique(Fk); remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return V[used], remap[Fk]


def clean_floating(V, F, keep=0.003, masks=None, H=None, tol=0.01, tiny=0.0005, log=None):
    """remove detached pieces only when they are proven junk. A piece other than the biggest is dropped if it is a crumb
    (under `tiny` of the biggest piece or under 8 triangles) or, when silhouettes are known, if most of it lies outside
    the silhouettes by more than `tol` (fraction of height). Without silhouettes only pieces under `keep` of the biggest go.
    Separate real parts (hair clip, ribbon, earring) inside the silhouettes stay, however small."""
    from scipy.sparse.csgraph import connected_components
    if len(F) == 0:
        return V, F
    n = len(V); e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    A = sparse.csr_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    nc, lab = connected_components(A, directed=False)
    if nc <= 1:
        return V, F
    fl = lab[F[:, 0]]; cnt = np.bincount(fl, minlength=nc); big = cnt.max(); drop = (cnt < 8) | (cnt < tiny * big)
    if masks and H:
        C = V[F].mean(1); out = _excess(C, masks, H) > tol
        frac = np.bincount(fl, weights=out.astype(float), minlength=nc) / np.maximum(cnt, 1)
        drop |= frac > 0.5
    else:
        drop |= cnt < keep * big
    drop[np.argmax(cnt)] = False
    if log:
        log(f"STATUS floating parts: {int(drop.sum())} removed, {int(((cnt > 0) & ~drop).sum()) - 1} small parts kept")
    return _keep_used(V, F, ~drop[fl])


def trim_outside(V, F, masks, H, tol=0.01, hard=None, max_removed=0.5, fill_max=400, log=None):
    """cut geometry that sticks out of the silhouettes without opening holes in the body.
    - a triangle is a candidate only if all its corners and its centre are outside some silhouette by more than `tol`
      (fraction of height, 1 % by default; the old fixed 2 px cut the real surface whenever a view was slightly off);
    - candidates are grouped into connected pieces; a piece is removed only if its worst point is outside by more than
      `hard` (2 x tol by default). Borderline pieces stay;
    - if removal would take more than `max_removed` of the mesh the silhouettes are probably wrong and nothing is cut;
    - the holes the cut opens (new open edges) are closed with a fan up to `fill_max` edges; larger ones are reported.
    masks: {angle: R x R bool} in the plugin frame"""
    if len(F) == 0 or not masks:
        return V, F
    hard = 2 * tol if hard is None else hard
    exc_v = _excess(V, masks, H); C = V[F].mean(1); exc_c = _excess(C, masks, H)
    cand = (exc_v[F].min(1) > tol) & (exc_c > tol)
    if not cand.any():
        return V, F
    lab, k = mc.component_labels(F[cand], len(V)); depth = np.zeros(k)
    np.maximum.at(depth, lab, np.maximum(exc_c[cand], exc_v[F[cand]].max(1)))
    cut = np.zeros(len(F), bool); cut[np.flatnonzero(cand)[depth[lab] > hard]] = True
    if cut.mean() > max_removed:
        if log:
            log(f"STATUS trim skipped: it would remove {100 * cut.mean():.0f} % of the mesh, check the silhouettes")
        return V, F
    old_keys, _ = mc.boundary_keys(F, len(V))
    V2, F2 = _keep_used(V, F, ~cut)
    # the open edges of the original mesh must be translated to the compacted vertex numbering
    used = np.unique(F[~cut]); remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    base_o = len(V) + 1; ok = old_keys; a0, b0 = ok // base_o, ok % base_o
    keep_e = (remap[a0] >= 0) & (remap[b0] >= 0); a1, b1 = remap[a0[keep_e]], remap[b0[keep_e]]
    old2 = np.minimum(a1, b1) * (len(V2) + 1) + np.maximum(a1, b1)
    V3, F3, filled, left = mc.fill_new_holes(V2, F2, old2, fill_max)
    if log:
        log(f"STATUS trim: {int(cut.sum())} triangles removed in {int((depth > hard).sum())} pieces, "
            f"{filled} holes closed, {left} large openings left")
    return V3, F3


def cleanup(V, F, B, H, p):
    """trim + floating-part removal with integrity reports before and after (the mesh must not get new holes)"""
    V, F = np.asarray(V), np.asarray(F); masks = getattr(B, "last_masks", {})
    before = mc.report(V, F); say("STATUS mesh before cleanup: " + mc.fmt(before))
    if p.get("trim", True):
        V, F = trim_outside(V, F, masks, H, log=say)
    if p.get("clean", True):
        V, F = clean_floating(V, F, masks=masks, H=H, log=say)
    after = mc.report(V, F); say("STATUS mesh after cleanup: " + mc.fmt(after))
    if after["open_edges"] > before["open_edges"]:
        say(f"STATUS warning: cleanup left {after['open_edges'] - before['open_edges']} more open edges than before")
    return V, F


def _mt(p):
    """file modification time: cached maps are reloaded after the file changes (e.g. Invert relief)"""
    try:
        return os.path.getmtime(p)
    except (OSError, TypeError):
        return 0.0


def say(*a):
    print(*a, flush=True)


def align_labels(lab, m, R, foot=0.045, top=0.045):
    """same transform as normals_volume.align (bbox of the normal map silhouette), nearest resampling"""
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    Hpx = R * (1 - foot - top); w = max(1, int(round((x1 - x0) * Hpx / (y1 - y0))))
    crop = np.asarray(Image.fromarray(lab[y0:y1, x0:x1]).resize((w, int(round(Hpx))), Image.NEAREST))
    cm = np.asarray(Image.fromarray((m[y0:y1, x0:x1] * 255).astype(np.uint8)).resize((w, int(round(Hpx))), Image.BILINEAR)) > 127
    legs = cm[-max(2, int(cm.shape[0] * 0.12)):]; cx = np.nonzero(legs)[1].mean() if legs.any() else w / 2
    out = np.zeros((R, R), np.uint8); oy = int(round(R * top)); ox = int(round(R / 2 - cx))
    xs_ = slice(max(0, ox), min(R, ox + w)); src = slice(max(0, -ox), max(0, -ox) + (xs_.stop - xs_.start))
    out[oy:oy + crop.shape[0], xs_] = crop[:, src]
    return out


class Builder:
    def __init__(self):
        self.c_load, self.c_int, self.c_hull, self.c_shift, self.c_lab = {}, {}, {}, {}, {}

    def labels(self, v, R):
        """part ids (0 = none) aligned like the view, or None"""
        lp = v.get("labels")
        if not lp or not os.path.exists(lp):
            return None
        key = (lp, v["path"], R)
        if key not in self.c_lab:
            raw = np.asarray(Image.open(lp).convert("RGBA")); ids = (raw[..., 0].astype(int) + 20) // 40   # id * 40 in R
            from normals_volume import load_view as _lv
            _, m0 = _lv(v["path"])
            self.c_lab[key] = align_labels(ids.astype(np.uint8), m0, R)
        return self.c_lab[key]

    # ---------------------------------------------------------------- inputs
    def load(self, v, R):
        key = (v["path"], v.get("fmt", "auto"), R, _mt(v["path"]))
        if key not in self.c_load:
            rgb, m = load_view(v["path"]); rgb, m = align(rgb, m, R)
            fmt = v.get("fmt", "auto")
            if fmt == "auto":
                n, _ = decode_normals(rgb, m)
            else:
                n = rgb / 127.5 - 1
                if fmt == "directx": n[..., 1] *= -1
                n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
            self.c_load[key] = (m, n.astype(np.float32))
        return self.c_load[key]

    def views_at(self, req, R, base_R):
        """aligned (m, n) per enabled view at resolution R, with the automatic + user shifts"""
        en = [v for v in req["views"] if v.get("enabled", True) and v.get("path") and os.path.exists(v["path"])]
        if len(en) < 2:
            raise ValueError("need at least 2 enabled views with existing normal maps")
        base = {float(v["angle"]): self.load(v, base_R) for v in en}
        skey = tuple(sorted((float(v["angle"]), v["path"]) for v in en)) + (base_R,)
        if skey not in self.c_shift:                                   # automatic horizontal alignment, once per set
            say("STATUS aligning views 2 %"); self.c_shift[skey] = refine_offsets(base, base_R)
        auto = self.c_shift[skey]; out = {}; self.keys = {}; self.labs = {}
        for v in en:
            a = float(v["angle"]); m, n = self.load(v, R) if R != base_R else base[a]
            dx = int(round((auto.get(a, 0) + v.get("dx", 0)) * R / base_R)); dy = int(round(v.get("dy", 0) * R / base_R))
            if dx or dy:
                m, n = np.roll(m, (dy, dx), axis=(0, 1)), np.roll(n, (dy, dx), axis=(0, 1))
            out[a] = (m, n); self.keys[a] = (v["path"], v.get("fmt", "auto"), dx, dy, v.get("labels") or "", _mt(v["path"]))
            lb = self.labels(v, R)
            if lb is not None and (dx or dy):
                lb = np.roll(lb, (dy, dx), axis=(0, 1))
            self.labs[a] = lb
        self.last_masks = {a: m for a, (m, _) in out.items()}          # for trimming the result to the silhouettes
        return out

    # ---------------------------------------------------------------- volume
    def volume(self, req, R, log=say):
        p = req["p"]; H = float(req["height"]); S = R * (1 - 2 * MARGIN) / H
        views = self.views_at(req, R, R)
        vkey = tuple(sorted((a,) + self.keys[a] for a in views)) + (R,)
        xs = np.arange(R) - R / 2 + 0.5
        ext = max(np.abs(xs[np.nonzero(m.any(0))[0]]).max() for m, _ in views.values()) + 3
        gx = np.arange(-ext, ext + 1, 1.0); X, Y = np.meshgrid(gx, gx, indexing="xy")
        P = {a: (X * math.cos(math.radians(a)) + Y * math.sin(math.radians(a)) + R / 2 - 0.5,
                 X * math.sin(math.radians(a)) - Y * math.cos(math.radians(a))) for a in views}
        rows_any = np.nonzero(np.any([m.any(1) for m, _ in views.values()], 0))[0]
        r0, r1 = rows_any.min() - 2, rows_any.max() + 3

        def hull_slab(s, e):
            ins = np.ones((e - s,) + X.shape, bool)
            for b, (m, _) in views.items():
                cb = np.round(P[b][0]).astype(int); ok = (cb >= 0) & (cb < R)
                ins &= m[s:e][:, np.clip(cb, 0, R - 1)] & ok[None]
            return ins

        todo = [(a, (a,) + self.keys[a] + (R, round(p["break_jump"], 3))) for a in views]
        todo = [(a, ik) for a, ik in todo if ik not in self.c_int]
        if todo:
            log(f"STATUS integrating {len(todo)} views in parallel")
            for (a, ik), res in zip(todo, pmap(lambda t: integrate(views[t[0]][0], views[t[0]][1], R, p["break_jump"], self.labs.get(t[0])), todo)):
                self.c_int[ik] = res
        depth = {}
        for a, (m, n) in views.items():
            hk = vkey + (a,)
            if hk not in self.c_hull:
                D = np.full((R, R), -np.inf); ci = np.clip(np.round(P[a][0]).astype(int), 0, R - 1); dep = P[a][1]
                for s in range(r0, r1, 64):
                    e = min(r1, s + 64); rr, yy, xx = np.nonzero(hull_slab(s, e))
                    np.maximum.at(D, (rr + s, ci[yy, xx]), dep[yy, xx])
                self.c_hull[hk] = D
            ik = (a,) + self.keys[a] + (R, round(p["break_jump"], 3))
            if ik not in self.c_int:
                log(f"STATUS integrating view {a:g}"); self.c_int[ik] = integrate(m, n, R, p["break_jump"], self.labs.get(a))
            ys, xs_, z, lab, nc = self.c_int[ik]
            hd = self.c_hull[hk][ys, xs_]; good = np.isfinite(hd)
            o = np.argsort(lab); d_ = (hd - z)[o]; l_ = lab[o]; g_ = good[o]; off = np.zeros(nc)
            u, st = np.unique(l_, return_index=True)
            for k, chunk, gk in zip(u, np.split(d_, st[1:]), np.split(g_, st[1:])):
                if gk.any(): off[k] = np.median(chunk[gk])
            zc = z + off[lab]; L = self.labs.get(a); parts = req.get("parts") or {}
            if L is not None and parts:                              # per-part settings
                ids = L[ys, xs_]
                mc = np.array([parts.get(str(i), {}).get("max_carve", p["max_carve"]) for i in range(256)])[ids]
                zc = np.where(mc > 0, np.maximum(zc, np.where(good, hd - mc * H * S, -np.inf)), zc)
            elif p["max_carve"] > 0:
                zc = np.maximum(zc, np.where(good, hd - p["max_carve"] * H * S, -np.inf))
            D = np.full((R, R), np.nan); D[ys, xs_] = np.minimum(zc, np.where(good, hd, np.inf))
            if L is not None and parts:                              # thinness measured inside each part
                dist = np.zeros((R, R)); thick = np.zeros((R, R), bool)
                for i in np.unique(L[m]):
                    pm = m & (L == i); di = ndimage.distance_transform_edt(pm); dist[pm] = di[pm]
                    q = parts.get(str(int(i)), {})
                    thick |= pm & (di > q.get("thin", p["thin"]) * R / 800)
                    if q.get("carve", False):
                        thick |= pm & (di > p["rim"] * R / 800)
            else:
                dist = ndimage.distance_transform_edt(m); thick = dist > p["thin"] * R / 800
                if p.get("use_carve", True):
                    rows_ = np.arange(R)[:, None]
                    thick |= (rows_ < R - 1 - (p["carve_above"] * H * S + R * MARGIN)) & (dist > p["rim"] * R / 800)
            depth[a] = (D, m & (n[..., 2] > 0.15) & thick, m)

        def sample(img, colf):
            c0 = np.floor(colf).astype(int); f = colf - c0
            return img[:, np.clip(c0, 0, R - 1)] * (1 - f) + img[:, np.clip(c0 + 1, 0, R - 1)] * f

        log("STATUS fusing views"); verts, faces, nv, BIG, PAD, soft = [], [], 0, 5.0, 4, p["soft"]
        for s in range(r0, r1, 96):
            e = min(r1, s + 98); ss, ee = max(0, s - PAD), min(R, e + PAD); fs = []
            for a, (D, trust, m) in depth.items():
                col, dep = P[a]; inside = (col >= 0) & (col <= R - 1)
                Ds = sample(np.nan_to_num(np.where(trust, D, np.nan)[ss:ee], nan=1e9), col)
                Ms = sample(m[ss:ee].astype(float), col) > 0.5
                fs.append(np.where(Ms & inside[None], np.clip(Ds - dep[None], -BIG, BIG), -BIG).astype(np.float32))
            Fs = np.stack(fs); mn = Fs.min(0)
            field = mn - soft * np.log(np.exp(-(Fs - mn) / soft).sum(0)) if soft > 0 else mn
            field = ndimage.gaussian_filter(field, 0.7)[s - ss:s - ss + (e - s)]
            if field.max() <= 0:
                continue
            v, fc, _, _ = measure.marching_cubes(np.pad(field, ((0, 0), (1, 1), (1, 1)), constant_values=-BIG), 0.0)
            v[:, 1:] -= 1; v[:, 0] += s; cz = v[fc].mean(1)[:, 0]
            fc = fc[(cz >= s + (0 if s == r0 else 1)) & (cz < s + 97)]
            verts.append(v); faces.append(fc + nv); nv += len(v)
        V = np.concatenate(verts); F = np.concatenate(faces)
        _, first, inv = np.unique(np.round(V * 1000).astype(np.int64), axis=0, return_index=True, return_inverse=True)
        V = V[first]; F = inv.reshape(-1)[F]; F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        W = np.c_[gx[0] + V[:, 2], gx[0] + V[:, 1], R - 1 - V[:, 0] - R * MARGIN] / S
        if (vnormals(W, F) * (W - W.mean(0))).sum() < 0:
            F = F[:, [0, 2, 1]]
        n_ = len(W); I = np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]; J = np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
        Adj = sparse.csr_matrix((np.ones(len(I)), (I, J)), shape=(n_, n_)); Adj.data[:] = 1; deg = np.asarray(Adj.sum(1)).ravel() + 1e-9
        for it in range(int(p["smooth"])):
            if it % 3 == 0:
                vn = vnormals(W, F); avg = Adj @ vn / deg[:, None]; avg /= np.linalg.norm(avg, axis=1, keepdims=True) + 1e-12
                w = np.clip(np.exp(-((1 - np.einsum("ij,ij->i", vn, avg)) / max(p["smooth_thr"], 1e-5)) ** 2), 0.08, 1)[:, None]
            for lam in (0.55, -0.58):
                W = W + lam * w * (Adj @ W / deg[:, None] - W)
        return W, F, views

    # ---------------------------------------------------------------- depth maps (method C2-C)
    def load_depth(self, v, R):
        """depth of a view in world units toward the camera, aligned exactly like its normal map; nan outside"""
        p = v.get("depth"); key = (p, v["path"], R, v.get("depth_range", 1.0), v.get("depth_invert", False), float(v.get("height", 1)), _mt(p), _mt(v["path"]))
        if key in self.c_load:
            return self.c_load[key]
        rgb0, m0 = load_view(v["path"]); h0, w0 = m0.shape
        if p.lower().endswith(".npy"):
            d = np.load(p).astype(np.float32)
        else:
            im = Image.open(p); a = np.asarray(im).astype(np.float32)
            if a.ndim == 3:
                a = a[..., 0]
            top = 65535.0 if a.max() > 255 else 255.0; u = a / top
            d = np.where(a > 0, (u - 0.5) * float(v.get("depth_range", 1.0)) * float(v.get("height", 1)), np.nan).astype(np.float32)
            if v.get("depth_invert"):
                d = -d
        if d.shape != (h0, w0):
            d = np.asarray(Image.fromarray(d.astype(np.float32), mode="F").resize((w0, h0), Image.NEAREST))
        good = m0 & np.isfinite(d)
        idx = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True); d = d[idx[0], idx[1]]
        ys, xs = np.nonzero(m0); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        Hpx = R * (1 - 2 * MARGIN); w = max(1, int(round((x1 - x0) * Hpx / (y1 - y0))))
        crop = np.asarray(Image.fromarray(d[y0:y1, x0:x1].astype(np.float32), mode="F").resize((w, int(round(Hpx))), Image.BILINEAR))
        cm = np.asarray(Image.fromarray((m0[y0:y1, x0:x1] * 255).astype(np.uint8)).resize((w, int(round(Hpx))), Image.BILINEAR)) > 127
        legs = cm[-max(2, int(cm.shape[0] * 0.12)):]; cx = np.nonzero(legs)[1].mean() if legs.any() else w / 2
        out = np.full((R, R), np.nan, np.float32); oy = int(round(R * MARGIN)); ox = int(round(R / 2 - cx))
        xs_ = slice(max(0, ox), min(R, ox + w)); src = slice(max(0, -ox), max(0, -ox) + (xs_.stop - xs_.start))
        out[oy:oy + crop.shape[0], xs_] = np.where(cm[:, src], crop[:, src], np.nan)
        self.c_load[key] = out
        return out

    def depth_mesh(self, req, N, log=say):
        import depth_fusion
        p = req["p"]; H = float(req["height"]); R = int(p.get("dn_maps", 1024))
        views = self.views_at(req, R, R); en = {float(v["angle"]): v for v in req["views"] if v.get("enabled", True) and v.get("path")}
        key = (tuple(sorted(self.keys.items())), N, R, bool(p.get("dn_smooth", True)), bool(p.get("dn_auto", True)), bool(p.get("dn_scale", True)), float(p.get("dn_max_carve", 0.0)),
               tuple(sorted((a, en[a].get("depth"), en[a].get("depth_range"), en[a].get("depth_invert"), _mt(en[a].get("depth") or "")) for a in views)))
        c = getattr(self, "_dn", None)
        if c and c[0] == key:
            return c[1], c[2]
        dv = {}
        for a, (m, n) in views.items():
            v = dict(en[a]); v["height"] = H
            if not v.get("depth") or not os.path.exists(v["depth"]):
                raise ValueError(f"view {a:g}: no depth map")
            d = self.load_depth(v, R); dx, dy = self.keys[a][2], self.keys[a][3]
            if dx or dy:
                d = np.roll(d, (dy, dx), axis=(0, 1))
            dv[a] = (np.where(m, d, np.nan), n, m & np.isfinite(d))
        log(f"STATUS depth fusion {N}^3 from {len(dv)} views 8 %")
        V, F, st = depth_fusion.fuse(dv, H, R, N, smooth=bool(p.get("dn_smooth", True)), auto_offset=bool(p.get("dn_auto", True)), auto_scale=bool(p.get("dn_scale", True)), max_carve=float(p.get("dn_max_carve", 0.0)), log=log)
        V = V.astype(np.float64); F = F.astype(np.int64); self._dn = (key, V, F); self.dn_stats = st
        return V, F

    # ---------------------------------------------------------------- fit maps
    def fit_maps(self, req, FR, R):
        views = self.views_at(req, FR, R)
        return {a: (m, n) for a, (m, n) in views.items()}


ROTS = None


def rotations():
    """the 24 proper rotations of the cube (any axis convention of a generator)"""
    global ROTS
    if ROTS is None:
        import itertools
        out = []
        for perm in itertools.permutations(range(3)):
            for sg in itertools.product((1, -1), repeat=3):
                M = np.zeros((3, 3))
                for i, j in enumerate(perm):
                    M[i, j] = sg[i]
                if np.linalg.det(M) > 0:
                    out.append(M)
        ROTS = out
    return ROTS


def silhouette_score(V, masks, H, R):
    S = R * (1 - 2 * MARGIN) / H; ious = []
    for a, m in masks.items():
        t = math.radians(a); right = np.array([math.cos(t), math.sin(t), 0])
        col = np.clip((V @ right * S + R / 2 - 0.5).astype(int), 0, R - 1); row = np.clip((R - 1 - (V[:, 2] * S + R * MARGIN)).astype(int), 0, R - 1)
        rep = np.zeros((R, R), bool); rep[row, col] = True
        rep = ndimage.binary_fill_holes(ndimage.binary_closing(rep, iterations=2))
        ious.append((rep & m).sum() / max(1, (rep | m).sum()))
    return float(np.mean(ious)), ious


def place(V, H):
    """feet at 0, height H, centred on the bounding box"""
    s = H / max(1e-9, np.ptp(V[:, 2])); V = V * s
    c = (V.max(0) + V.min(0)) / 2; return V - [c[0], c[1], V[:, 2].min()]


def align_start(B, req, raw, log=say):
    """orientation (24 rotations), height and offset of a generated mesh from the view silhouettes"""
    with np.load(raw) as d:
        V0 = d["V"].astype(np.float64); F = d["F"].astype(np.int64)
    H = float(req["height"]); R = 256
    masks = {a: m for a, (m, _) in B.views_at(req, R, R).items()}
    sub = V0[np.random.default_rng(0).choice(len(V0), min(len(V0), 60000), replace=False)]
    best = None
    for M in rotations():
        sc, _ = silhouette_score(place(sub @ M.T, H), masks, H, R)
        if best is None or sc > best[0]:
            best = (sc, M)
    M = best[1]; Vb = place(sub @ M.T, H); bs = (best[0], 1.0, 0.0, 0.0)
    log(f"STATUS orientation found, silhouette IoU {best[0]:.3f}")
    for _ in range(2):                                               # fine scale / offset search
        for s_ in (0.94, 0.97, 1.0, 1.03, 1.06):
            for dx in np.linspace(-0.04, 0.04, 5) * H:
                for dy in np.linspace(-0.04, 0.04, 5) * H:
                    Vt = Vb * [s_, s_, 1.0] + [dx, dy, 0]
                    sc, _ = silhouette_score(Vt, masks, H, R)
                    if sc > bs[0]:
                        bs = (sc, s_, dx, dy)
        Vb = Vb * [bs[1], bs[1], 1.0] + [bs[2], bs[3], 0]; bs = (bs[0], 1.0, 0.0, 0.0)
    # same transform on the full mesh
    Vf = place(V0 @ M.T, H); Vs = place(sub @ M.T, H)
    k = np.linalg.lstsq(np.c_[Vs[:, :2], np.ones(len(Vs))], Vb[:, :2], rcond=None)[0]   # recover xy scale + offset
    Vf[:, :2] = np.c_[Vf[:, :2], np.ones(len(Vf))] @ k
    sc, per = silhouette_score(Vf[::max(1, len(Vf) // 60000)], masks, H, R)
    return Vf, F, {"silhouette_iou": round(sc, 3), "per_view": {str(int(a)): round(x, 3) for a, x in zip(masks, per)}}


def sil_fit(B, req, V, F, log=say):
    """fit a mesh (neural start or volume) to the view silhouettes when sil_iters > 0; cached per mesh + settings"""
    import hashlib, silhouette_fit as sf
    p = req["p"]; H = float(req["height"]); its = int(p.get("sil_iters", 0))
    if its <= 0:
        B.sil_iou = None; return V, F
    SR = int(p.get("sil_res", 512)); masks = {a: m for a, (m, _) in B.views_at(req, SR, SR).items()}
    h = hashlib.md5(np.ascontiguousarray(V[::53], np.float32).tobytes() + np.int64(len(V)).tobytes()).hexdigest()
    key = (h, tuple(sorted(B.keys.items())), its, p.get("sil_step"), p.get("sil_spread"), p.get("sil_smooth"), p.get("sil_max"), SR)
    c = getattr(B, "_sil", None)
    if c and c[0] == key:
        return c[1].copy(), F
    per = sf.iou(masks, V, F, H); before = np.mean(list(per.values()))
    log("STATUS silhouette IoU per view: " + ", ".join(f"{int(a)}:{v:.2f}" for a, v in sorted(per.items())))
    med = float(np.median(list(per.values()))); bad = [a for a, v in per.items() if v < med - 0.15]
    if bad and len(bad) < len(per):                          # a view that disagrees with the rest would drag the mesh the wrong way
        log("STATUS silhouette fit ignores views that do not match the mesh: " + ", ".join(str(int(a)) for a in sorted(bad)))
        masks = {a: m for a, m in masks.items() if a not in bad}
    V, sc = sf.conform(V, F, masks, H, iters=its, step=float(p.get("sil_step", 0.8)), spread=int(p.get("sil_spread", 40)),
                       smooth=int(p.get("sil_smooth", 2)), max_shift=float(p.get("sil_max", 0.02)) * H, log=log)
    after = float(np.mean(list(sc.values()))); log(f"STATUS silhouette IoU {before:.3f} -> {after:.3f}")
    B._sil = (key, V.copy()); B.sil_iou = (round(float(before), 3), round(after, 3))
    return V, F


def start_mesh(B, req, log=say):
    with np.load(req["start"]) as d:
        V, F = d["V"].astype(np.float64), d["F"].astype(np.int64)
    return sil_fit(B, req, V, F, log)


def live():
    q = queue.Queue(); B = Builder()

    def reader():
        for line in sys.stdin:
            line = line.strip()
            if line:
                q.put(line)
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()
    say("STATUS ready")
    while True:
        item = q.get()
        if item is None:
            return
        while not q.empty():                                          # newest request wins
            nxt = q.get()
            if nxt is None:
                return
            item = nxt
        try:
            req = json.loads(item)
        except Exception:
            continue
        if req.get("cmd") == "quit":
            return
        rid = req.get("id", 0)
        try:
            t0 = time.time(); p = req["p"]; R = int(req.get("res", 320)); H = float(req["height"])
            if req.get("depth_mode"):
                V, F = B.depth_mesh(req, int(p.get("dn_res", 128)))
            elif req.get("start") and os.path.exists(req["start"]):
                V, F = start_mesh(B, req)
            else:
                V, F, _ = B.volume(req, R); V, F = sil_fit(B, req, V, F)
            st = {"volume_s": round(time.time() - t0, 1), "triangles": int(len(F))}
            if getattr(B, "sil_iou", None):
                st["silhouette_iou"] = B.sil_iou[1]
            its = int(p.get("fit_iters", 0)); maps = None
            if its > 0 or p.get("diag") or req.get("aligned_dir"):
                maps = B.fit_maps(req, int(req.get("fit_res", 512)), R)
            ad = req.get("aligned_dir")
            if ad:                                                    # aligned maps for the viewport overlay
                os.makedirs(ad, exist_ok=True); sig = json.dumps(sorted(map(list, B.keys.items())))
                if getattr(B, "_ad_sig", None) != sig:
                    for a, (m, n) in maps.items():
                        img = np.dstack([((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8), (m * 160).astype(np.uint8)])
                        Image.fromarray(img).save(os.path.join(ad, f"n_{int(a)}.png"))
                    B._ad_sig = sig
            if its > 0:
                for k in range(its):
                    if not q.empty():                                 # a newer request arrived: drop this one
                        break
                    lam = p["lam0"] + (p["lam1"] - p["lam0"]) * (k / max(1, its - 1))
                    say(f"STATUS fit {k + 1}/{its}")
                    V, F, s_ = mvf.fit_step(V, F, maps, H, lam, p["mu"], p["agree"], p["relax"], max_step=float(p.get("fit_max", 0.005)) * H)
                    st["faces_vs_maps_deg"] = s_["median_face_vs_maps_deg"]
                st["folded_pct"] = round(100 * mvf.folded_share(V, F), 2)
            V, F = cleanup(V, F, B, H, p)
            rep = mc.report(V, F); st["open_edges"] = rep["open_edges"]; st["parts"] = rep["components"]
            extra = {}
            if p.get("diag"):
                ve, vs = mvf.observe(V, F, maps, H, p["agree"], vertex=True); extra = {"err": ve, "seen": vs}
                st["unseen_pct"] = round(100 * float((vs < 0.5).mean()), 1)
            st["total_s"] = round(time.time() - t0, 1)
            out = req["out"]; tmp = out + ".tmp.npz"
            np.savez(tmp, V=V.astype(np.float32), F=F.astype(np.int32), **extra); os.replace(tmp, out)
            say(f"MESH {rid} {json.dumps(st)}")
        except Exception as e:
            say(f"ERR {rid} {type(e).__name__}: {e}".replace("\n", " ")[:400])
            traceback.print_exc(file=sys.stderr)


def final(job_path):
    req = json.load(open(job_path, encoding="utf-8")); B = Builder(); R = int(req.get("res", 800))
    wd = req["work"]; al = os.path.join(wd, "aligned"); os.makedirs(al, exist_ok=True)
    if req.get("align_start"):                                       # generated mesh -> view frame
        V, F, info = align_start(B, req, req["align_start"])
        np.savez(req["start_out"], V=V.astype(np.float32), F=F.astype(np.int32))
        say("START " + json.dumps(info)); return
    if req.get("depth_mode"):
        V, F = B.depth_mesh(req, int(req["p"].get("dn_res", 256))); say("DEPTH " + json.dumps(B.dn_stats))
    elif req.get("start") and os.path.exists(req["start"]):
        V, F = start_mesh(B, req)
    else:
        V, F, _ = B.volume(req, R); V, F = sil_fit(B, req, V, F)
    if getattr(B, "sil_iou", None):
        say(f"silhouette IoU {B.sil_iou[0]} -> {B.sil_iou[1]}")
    n0 = len(F)
    V, F = cleanup(V, F, B, float(req["height"]), req["p"])
    if n0 != len(F):
        say(f"STATUS outside / floating parts: {n0} -> {len(F)} triangles")
    np.savez(os.path.join(wd, "underlay.npz"), V=V.astype(np.float32), F=F.astype(np.int32))
    say(f"underlay.obj: {len(V)} verts, {len(F)} tris")
    for a, (m, n) in ({} if req.get("no_fit_maps") else B.fit_maps(req, int(req.get("fit_res", 2048)), R)).items():
        out = np.dstack([((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8), (m * 255).astype(np.uint8)])
        Image.fromarray(out).save(os.path.join(al, f"n_{int(a)}.png"))
    say("PROGRESS 1 1 volume done")


if __name__ == "__main__":
    if "--final" in sys.argv:
        final(sys.argv[sys.argv.index("--final") + 1])
    else:
        live()
