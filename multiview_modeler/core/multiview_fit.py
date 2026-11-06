"""multiview_fit.py - fit a mesh to N normal maps at once (numpy + scipy, CPU).

  python multiview_fit.py IN.npz OUT.npz --view 0:aligned/n_0.png --view 90:aligned/n_90.png ... --height 2.2
         [--iters 8] [--lam 0.3,0.3,0.3,0.3,0.2,0.15,0.1,0.1] [--subdivide-above 0.56] [--mu 0.05] [--report fit.json]

Maps must be in the frame written by normals_volume.py --aligned-out (square, feet at 4.5 % from the bottom,
height = 91 % of the image, character centred, OpenGL normals, alpha = silhouette). Mesh: Z up, faces -Y, feet at 0.

Per iteration: E(V) = sum_faces sum_edges w (n_f . (Vj - Vi))^2              (faces follow the normals of all views)
                    + LAM^2 sum_edges |d_i - d_j|^2                          (neighbours move together: no fold-overs)
                    + MU |d|^2,                                              (small steps)
n_f = robust consensus of the views that see the face (views >25 deg off the consensus are dropped);
then faces that would flip are rolled back and vertices are relaxed along the surface (light remesh).
Every iteration runs in its own process (memory is returned to the OS); progress lines: "PROGRESS i n text".
"""
import argparse, json, math, os, shutil, subprocess, sys, time
import numpy as np
from par import pmap
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import LinearOperator, lsqr

MARGIN = 0.045


def load_npz(p):
    with np.load(p) as d:                                            # closes the file (Windows locks open files)
        return d["V"].astype(np.float64), d["F"].astype(np.int64)


def fnorm(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); return fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)


def vnorm(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); vn = np.zeros_like(V)
    for k in range(3): np.add.at(vn, F[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def folded_share(V, F):
    fn = fnorm(V, F); e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; e.sort(1); fid = np.tile(np.arange(len(F)), 3)
    k = e[:, 0] * len(V) + e[:, 1]; o = np.argsort(k); k, fid = k[o], fid[o]; same = k[1:] == k[:-1]
    return float(((fn[fid[1:][same]] * fn[fid[:-1][same]]).sum(1) < -0.3).mean())


def subdivide(V, F, zmin):
    """1 -> 4 split of the faces above zmin (detail zone); shared midpoints"""
    sel = V[F][:, :, 2].max(1) > zmin; Fs, Fk = F[sel], F[~sel]
    e = np.r_[Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]]; e.sort(1); key = e[:, 0] * len(V) + e[:, 1]
    uk, inv = np.unique(key, return_inverse=True); a, b = uk // len(V), uk % len(V)
    mi = len(V) + inv.reshape(3, -1).T; v0, v1, v2 = Fs.T; m01, m12, m20 = mi.T
    F2 = np.r_[Fk, np.c_[v0, m01, m20], np.c_[m01, v1, m12], np.c_[m20, m12, v2], np.c_[m01, m12, m20]]
    return np.r_[V, (V[a] + V[b]) / 2], F2


def load_maps(specs):
    maps = {}
    for s in specs:
        ang, p = s.split(":", 1); im = np.asarray(Image.open(p).convert("RGBA")).astype(np.float32)
        n = im[..., :3] / 127.5 - 1; n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        maps[float(ang)] = (im[..., 3] > 127, n)
    return maps


def one_iteration(a):
    V, F = load_npz(a.inp); maps = load_maps(a.view)
    V, F, st = fit_step(V, F, maps, a.height, a.lam_now, a.mu, a.agree, a.relax, max_step=(a.max_step * a.height if a.max_step else None))
    np.savez(a.out, V=V.astype(np.float32), F=F.astype(np.int32))
    return st


def observe(V, F, maps, H, agree_deg=25.0, vertex=False):
    """robust multi-view target normal per face (+ weight, coverage); vertex=True -> per-vertex error / seen count"""
    nF = len(F)
    fn = fnorm(V, F); C = V[F].mean(1)

    def one(item):
        ang, (m, nm) = item
        R = m.shape[0]; S = R * (1 - 2 * MARGIN) / H
        t = math.radians(ang); right = np.array([math.cos(t), math.sin(t), 0]); cam = np.array([math.sin(t), -math.cos(t), 0])
        col = C @ right * S + R / 2 - 0.5; row = R - 1 - (C[:, 2] * S + R * MARGIN); dep = C @ cam * S
        ci = np.clip(np.round(col).astype(int), 0, R - 1); ri = np.clip(np.round(row).astype(int), 0, R - 1)
        zb = np.full((R, R), -np.inf, np.float32); np.maximum.at(zb, (ri, ci), dep.astype(np.float32))
        holes = ~np.isfinite(zb); zf = ndimage.maximum_filter(zb, 3); zb[holes] = zf[holes]
        facing = fn @ cam
        tol = 1.5 + np.clip(2.5 * np.sqrt(np.clip(1 - facing ** 2, 0, 1)) / np.maximum(facing, 0.2), 0, 6)
        vis = (dep > zb[ri, ci] - tol) & m[ri, ci] & (facing > 0.05)
        c0 = np.clip(np.floor(col).astype(int), 0, R - 2); r0 = np.clip(np.floor(row).astype(int), 0, R - 2)
        fx = (col - c0)[:, None]; fy = (row - r0)[:, None]
        nw = nm[r0, c0] * (1 - fx) * (1 - fy) + nm[r0, c0 + 1] * fx * (1 - fy) + nm[r0 + 1, c0] * (1 - fx) * fy + nm[r0 + 1, c0 + 1] * fx * fy
        nwld = nw[:, :1] * right + nw[:, 1:2] * np.array([0, 0, 1.0]) + nw[:, 2:3] * cam
        return ((nwld / (np.linalg.norm(nwld, axis=1, keepdims=True) + 1e-12)).astype(np.float32),
                np.where(vis, np.clip(facing, 0, 1) ** 2, 0).astype(np.float32))
    res = pmap(one, maps.items()); O = np.stack([r_[0] for r_ in res]); W = np.stack([r_[1] for r_ in res]); del res
    for _ in range(2):
        mean = (O * W[..., None]).sum(0); mean /= np.linalg.norm(mean, axis=1, keepdims=True) + 1e-12
        agree = np.einsum("vfi,fi->vf", O, mean) > math.cos(math.radians(agree_deg))
        W = W * np.where(agree | (W.sum(0) == 0)[None], 1, 0.02)
    mean = (O * W[..., None]).sum(0); ws = W.sum(0); del O
    tgt = np.where(ws[:, None] > 1e-6, mean / (np.linalg.norm(mean, axis=1, keepdims=True) + 1e-12), fn)
    w = np.clip(ws, 0, 1); cov = float((ws > 1e-6).mean())
    if vertex:                                                       # diagnostics for the viewport colours
        seen = (W > 0.02).sum(0).astype(np.float32); del mean, W
        err = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fn, tgt), -1, 1))).astype(np.float32)
        err[ws <= 1e-6] = 0
        ve = np.zeros(len(V), np.float32); vs = np.zeros(len(V), np.float32); vc = np.zeros(len(V), np.float32)
        for k in range(3):
            np.add.at(ve, F[:, k], err); np.add.at(vs, F[:, k], seen); np.add.at(vc, F[:, k], 1)
        return ve / np.maximum(vc, 1), vs / np.maximum(vc, 1)
    del mean, W
    return tgt, w, cov, fn


def fit_step(V, F, maps, H, lam, mu=0.05, agree_deg=25.0, relax=0.3, max_step=None):
    """one fit iteration in memory (used by the CLI and by the live worker); returns V, F, stats.
    max_step: no vertex moves farther than this (scene units) in one iteration"""
    t0 = time.time()
    if (fnorm(V, F) * (V[F].mean(1) - V.mean(0))).sum() < 0: F = F[:, [0, 2, 1]]
    nF, nV = len(F), len(V)
    tgt, w, cov, fn = observe(V, F, maps, H, agree_deg)
    # --- solve
    E_i = np.r_[F[:, 0], F[:, 1], F[:, 2]]; E_j = np.r_[F[:, 1], F[:, 2], F[:, 0]]; E_f = np.tile(np.arange(nF), 3)
    wr = np.where(w > 1e-6, 0.3 + 0.7 * w, 0.15)[E_f]
    n = (tgt[E_f] * wr[:, None]).astype(np.float32); data = np.c_[n, -n].ravel(); del n, wr
    ej, ei = (3 * E_j).astype(np.int32), (3 * E_i).astype(np.int32)
    cols = np.c_[ej, ej + 1, ej + 2, ei, ei + 1, ei + 2].ravel(); del ej, ei
    A = sparse.csr_matrix((data, cols, np.arange(0, 18 * nF + 1, 6, dtype=np.int64)), shape=(3 * nF, 3 * nV)); del data, cols
    ue = np.unique(np.sort(np.c_[E_i, E_j], 1), axis=0); nE = len(ue)
    Lm = sparse.csr_matrix((np.r_[np.ones(nE, np.float32), -np.ones(nE, np.float32)],
                            (np.r_[np.arange(nE), np.arange(nE)], np.r_[ue[:, 0], ue[:, 1]])), shape=(nE, nV))
    el = np.linalg.norm(V[ue[:, 0]] - V[ue[:, 1]], axis=1).mean(); del ue, E_i, E_j, E_f
    r = -(A @ V.ravel()); k1 = 3 * nF; LAM = lam
    op = LinearOperator((k1 + 3 * nE, 3 * nV), dtype=np.float64,
                        matvec=lambda x: np.r_[A @ x, LAM * (Lm @ x.reshape(-1, 3)).ravel()],
                        rmatvec=lambda y: A.T @ y[:k1] + LAM * (Lm.T @ y[k1:].reshape(-1, 3)).ravel())
    d = lsqr(op, np.r_[r, np.zeros(3 * nE)], damp=math.sqrt(mu), iter_lim=120, atol=1e-7, btol=1e-7)[0].reshape(-1, 3)
    del A, r
    if max_step:
        nd = np.linalg.norm(d, axis=1); d *= np.minimum(1.0, max_step / np.maximum(nd, 1e-12))[:, None]
    rolled = 0
    for _ in range(4):
        bad = np.einsum("ij,ij->i", fnorm(V + d, F), fn) < 0.2
        if not bad.any(): break
        vb = np.unique(F[bad]); d[vb] *= 0.3; rolled += len(vb)
    V = V + d
    Adj = abs(Lm).T @ abs(Lm); Adj.setdiag(0); Adj.eliminate_zeros(); deg = np.asarray(Adj.sum(1)).ravel() + 1e-9
    for _ in range(3 if relax > 0 else 0):                           # tangential relaxation = light remesh
        vn = vnorm(V, F); c = Adj @ V / deg[:, None] - V
        V = V + relax * (c - vn * np.einsum("ij,ij->i", c, vn)[:, None])
    err = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fnorm(V, F), tgt), -1, 1)))
    return V, F, {"observed_faces": round(cov, 3), "median_face_vs_maps_deg": round(float(np.median(err)), 2),
            "mean_step_edges": round(float(np.linalg.norm(d, axis=1).mean() / el), 3), "rolled_back_vertices": int(rolled),
            "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--view", action="append", required=True, help="angle:aligned_normal_map.png (0 = front, 90 = left side)")
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--lam", default="0.3,0.3,0.3,0.3,0.2,0.15,0.1,0.1", help="fold protection per iteration (last value repeats)")
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--agree", type=float, default=25.0, help="views more than this many degrees off the consensus are dropped")
    ap.add_argument("--relax", type=float, default=0.3, help="tangential relaxation (light remesh) strength, 0 = off")
    ap.add_argument("--subdivide-above", type=float, default=None, help="height fraction: 1->4 split of the faces above (detail zone)")
    ap.add_argument("--max-step", type=float, default=0.005, help="largest move of any vertex per iteration, fraction of the height (0 = no limit)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--_one", type=float, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._one is not None:                                           # child process: one iteration
        a.lam_now = a._one; print("RESULT " + json.dumps(one_iteration(a)), flush=True); return
    lams = [float(x) for x in a.lam.split(",")]
    work = a.out + ".work.npz"
    V, F = load_npz(a.inp)
    if a.subdivide_above is not None:
        V, F = subdivide(V, F, a.subdivide_above * a.height); print(f"PROGRESS 0 {a.iters} subdivided: {len(F)} triangles", flush=True)
    np.savez(work, V=V.astype(np.float32), F=F.astype(np.int32)); del V, F
    log = []
    for it in range(a.iters):
        lam = lams[min(it, len(lams) - 1)]
        cmd = [sys.executable, os.path.abspath(__file__), work, work] + sum([["--view", v] for v in a.view], []) + \
              ["--height", str(a.height), "--mu", str(a.mu), "--agree", str(a.agree), "--relax", str(a.relax), "--max-step", str(a.max_step), "--_one", str(lam)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stderr[-2000:], file=sys.stderr); sys.exit(f"iteration {it + 1} failed (out of memory? lower the map resolution or --subdivide-above)")
        res = json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT ")][-1][7:]); res["lam"] = lam; log.append(res)
        prev = a.out + ".preview.npz"; shutil.copyfile(work, prev + ".tmp.npz"); os.replace(prev + ".tmp.npz", prev)   # for live preview
        print(f"PROGRESS {it + 1} {a.iters} faces vs maps {res['median_face_vs_maps_deg']} deg, {res['seconds']} s", flush=True)
    os.replace(work, a.out)
    V, F = load_npz(a.out)
    rep = {"iterations": log, "triangles": int(len(F)), "folded_pairs_share": round(folded_share(V, F), 4),
           "note": "median_face_vs_maps should fall every iteration; folded_pairs_share > 0.01 means raise --lam"}
    if a.report: json.dump(rep, open(a.report, "w"), indent=1)
    print("DONE " + json.dumps({k: rep[k] for k in ("triangles", "folded_pairs_share")}), flush=True)


if __name__ == "__main__":
    main()
