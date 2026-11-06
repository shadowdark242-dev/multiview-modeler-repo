# SPDX-License-Identifier: GPL-3.0-or-later
"""mesh_check.py - mesh integrity helpers (numpy only, so Blender's own Python can import it too).

report(V, F)               -> triangles, open (boundary) edges, non-manifold edges, connected components, boundary loops
fill_new_holes(V, F, ...)  -> close the small holes that an earlier deletion opened (centroid fan)
component_labels(F, n)     -> connected component label of every triangle
"""
import numpy as np


def _edge_keys(F, base):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1).astype(np.int64)
    return e[:, 0] * base + e[:, 1]


def boundary_keys(F, nv):
    """undirected keys of edges used by exactly one triangle (open edges), plus the base used for the keys"""
    if len(F) == 0:
        return np.zeros(0, np.int64), nv + 1
    base = nv + 1; uk, cnt = np.unique(_edge_keys(F, base), return_counts=True)
    return uk[cnt == 1], base


def vertex_components(edges, nv):
    """label of each vertex's connected component (label propagation with pointer jumping)"""
    lab = np.arange(nv)
    if len(edges) == 0:
        return lab
    a0, b0 = edges[:, 0], edges[:, 1]
    while True:
        a, b = lab[a0], lab[b0]; m = np.minimum(a, b); new = lab.copy()
        np.minimum.at(new, a, m); np.minimum.at(new, b, m)
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def component_labels(F, nv):
    """component label (0..k-1) of every triangle, k"""
    if len(F) == 0:
        return np.zeros(0, np.int64), 0
    lab = vertex_components(np.concatenate([F[:, [0, 1]], F[:, [1, 2]]]), nv)
    u, inv = np.unique(lab[F[:, 0]], return_inverse=True)
    return inv, len(u)


def boundary_loops(F, keys, base):
    """closed loops (lists of vertex indices) made of the open edges listed in `keys`; chains that do not close are dropped"""
    if len(keys) == 0:
        return []
    ks = set(int(k) for k in keys); nxt = {}
    for t in F:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = int(min(a, b)) * base + int(max(a, b))
            if k in ks:
                nxt.setdefault(int(a), []).append(int(b))
    loops = []
    for start in list(nxt):
        while nxt.get(start):
            loop = [start]; cur = nxt[start].pop(); ok = True
            while cur != start:
                loop.append(cur)
                if not nxt.get(cur):
                    ok = False; break
                cur = nxt[cur].pop()
            if ok and len(loop) >= 3:
                loops.append(loop)
    return loops


def report(V, F):
    nv = len(V); F = np.asarray(F)
    if len(F) == 0:
        return {"triangles": 0, "open_edges": 0, "nonmanifold_edges": 0, "components": 0, "loops": 0}
    base = nv + 1; uk, cnt = np.unique(_edge_keys(F, base), return_counts=True)
    bk = uk[cnt == 1]
    return {"triangles": int(len(F)), "open_edges": int(len(bk)), "nonmanifold_edges": int((cnt > 2).sum()),
            "components": component_labels(F, nv)[1], "loops": len(boundary_loops(F, bk, base)) if len(bk) < 200000 else -1}


def fmt(r):
    return f"{r['triangles']} tris, {r['open_edges']} open edges, {r['components']} parts"


def fill_new_holes(V, F, old_keys, max_edges=400):
    """close holes bounded only by edges that were NOT open before (`old_keys` = boundary_keys of the mesh before the
    deletion); each hole of at most max_edges edges gets a centroid fan. Returns V, F, filled, left"""
    nv = len(V)
    if len(F) == 0:
        return V, F, 0, 0
    bk, base = boundary_keys(F, nv)
    new = np.setdiff1d(bk, old_keys)
    if len(new) == 0 or len(new) > 200000:
        return V, F, 0, 0
    loops = boundary_loops(F, new, base); add_v = []; add_f = []; filled = 0; left = 0
    for lp in loops:
        if len(lp) > max_edges:
            left += 1; continue
        c = V[lp].mean(0); ci = nv + len(add_v); add_v.append(c); filled += 1
        for i in range(len(lp)):
            a, b = lp[i], lp[(i + 1) % len(lp)]
            add_f.append((b, a, ci))                  # opposite direction to the open edge a->b: consistent winding
    if not filled:
        return V, F, 0, left
    return np.vstack([V, np.array(add_v, V.dtype)]), np.vstack([F, np.array(add_f, F.dtype)]), filled, left
