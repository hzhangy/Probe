#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_w_op_w2_richardson.py

OP-W2: Richardson extrapolation to extract a_2 to O(h^4) accuracy.

Setup:
  On a mesh with spacing h, the cotangent Laplacian has systematic
  O(h^2) discretization error. Any extracted spectral quantity Q(h)
  satisfies
      Q(h) = Q_exact + C h^2 + O(h^4).

  Combining results at h and h/2 gives
      Q_richardson = (4 Q(h/2) - Q(h)) / 3
  which cancels the O(h^2) term.

Mesh:
  Octahedron geodesic subdivision. Each level halves the edge length
  and quadruples the face count: N_faces = 8, 32, 128, 512, 2048.

Extraction:
  Voronoi mass matrix M. Generalized eigenvalue problem L v = lambda M v.
  Heat kernel: K(t) = sum exp(-t lambda_k).
  Window: t in [c_uv / lambda_max, c_ir / lambda_min_nonzero].
  Fit t * K(t) = c_0 + c_1 * t. Then
      a_0 / (4 pi) = c_0,  a_2 / (4 pi) = c_1.

Targets:
  a_0 / (4 pi) = 1.000000
  a_2 / (4 pi) = 1/3 = 0.333333
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor, as_completed


# ── Octahedron + subdivision ──
def init_octahedron():
    verts = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0],
        [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=float)
    faces = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ]
    return verts, faces


def subdivide(verts, faces):
    edge_mid = {}
    new_verts = list(verts)
    new_faces = []

    def get_mid(i, j):
        key = (min(i, j), max(i, j))
        if key not in edge_mid:
            m = (new_verts[i] + new_verts[j]) / 2.0
            m /= np.linalg.norm(m)
            edge_mid[key] = len(new_verts)
            new_verts.append(m)
        return edge_mid[key]

    for f in faces:
        v0, v1, v2 = f
        m01 = get_mid(v0, v1)
        m12 = get_mid(v1, v2)
        m20 = get_mid(v2, v0)
        new_faces.extend([[v0, m01, m20], [v1, m12, m01],
                          [v2, m20, m12], [m01, m12, m20]])
    return np.array(new_verts), np.array(new_faces)


# ── Cotangent Laplacian + Voronoi mass ──
def build_L_and_M(verts, faces):
    N = len(verts)
    rows, cols, vals = [], [], []
    diag_L = np.zeros(N)
    diag_M = np.zeros(N)

    for f in faces:
        v0, v1, v2 = f
        p0, p1, p2 = verts[v0], verts[v1], verts[v2]

        def cot(pa, pb, pc):
            e1 = pb - pa
            e2 = pc - pa
            cross = np.linalg.norm(np.cross(e1, e2))
            if cross < 1e-14:
                return 0.0
            return np.dot(e1, e2) / cross

        c0 = cot(p0, p1, p2)
        c1 = cot(p1, p2, p0)
        c2 = cot(p2, p0, p1)

        for i, j, w in [(v1, v2, 0.5*c0), (v0, v2, 0.5*c1), (v0, v1, 0.5*c2)]:
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([-w, -w])
            diag_L[i] += w
            diag_L[j] += w

        e12 = np.linalg.norm(p1 - p2)**2
        e02 = np.linalg.norm(p0 - p2)**2
        e01 = np.linalg.norm(p0 - p1)**2
        A0 = (1.0 / 8.0) * (e01 * c2 + e02 * c1)
        A1 = (1.0 / 8.0) * (e01 * c2 + e12 * c0)
        A2 = (1.0 / 8.0) * (e02 * c1 + e12 * c0)
        diag_M[v0] += A0
        diag_M[v1] += A1
        diag_M[v2] += A2

    rows.extend(list(range(N)))
    cols.extend(list(range(N)))
    vals.extend(list(diag_L))

    L = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    L = 0.5 * (L + L.T)
    M = sp.diags(diag_M)
    return L.tocsr(), M.tocsr(), diag_M


# ── Heat kernel with generalized eigenvalues ──
def compute_a0_a2(evals, c_uv=8.0, c_ir=0.3, n_pts=40):
    """
    K(t) = sum exp(-t * lambda_k)
    Fit t*K(t) = c0 + c1*t in window t in [c_uv/lambda_max, c_ir/lambda_min_nz].
    c0 = a0/(4 pi), c1 = a2/(4 pi).
    """
    ev = np.sort(evals)
    ev = ev[ev > 1e-12]
    if len(ev) < 20:
        return float("nan"), float("nan")

    lam_max = ev[-1]
    lam_min_nz = ev[0]

    t_lo = c_uv / lam_max
    t_hi = c_ir / lam_min_nz

    if t_hi <= t_lo * 2:
        return float("nan"), float("nan")

    t_arr = np.logspace(np.log10(t_lo), np.log10(t_hi), n_pts)
    K = np.exp(-np.outer(t_arr, ev)).sum(axis=1)
    tK = t_arr * K

    # Linear fit: tK = c0 + c1 * t
    c1, c0 = np.polyfit(t_arr, tK, 1)
    return c0, c1


# ── Worker ──
def worker(n_sub):
    verts, faces = init_octahedron()
    for _ in range(n_sub):
        verts, faces = subdivide(verts, faces)
    Nf = len(faces)

    L, M, diag_M = build_L_and_M(verts, faces)

    # Generalized eig: L v = lambda M v
    M_inv_sqrt = sp.diags(1.0 / np.sqrt(diag_M))
    M_sqrt = sp.diags(np.sqrt(diag_M))
    L_sym = M_inv_sqrt @ L @ M_inv_sqrt
    L_sym = 0.5 * (L_sym + L_sym.T)

    # Full diagonalization if small, sparse otherwise
    if Nf <= 3000:
        evals = la.eigvalsh(L_sym.toarray())
    else:
        from scipy.sparse.linalg import eigsh
        k = min(Nf - 2, 4000)
        evals = eigsh(L_sym, k=k, sigma=-0.01, which='LM',
                      return_eigenvectors=False)
        evals = np.sort(evals)

    c0, c1 = compute_a0_a2(evals)

    return {
        'n_sub': n_sub,
        'N_faces': Nf,
        'N_vert': len(verts),
        'c0': c0,
        'c1': c1,
    }


def main():
    print("=" * 92)
    print("  OP-W2: Richardson extrapolation for a_2 to O(h^4) accuracy")
    print("=" * 92)
    print()

    n_levels = 6  # N_faces = 8, 32, 128, 512, 2048, 8192

    results = {}
    with ProcessPoolExecutor(max_workers=n_levels) as ex:
        futures = {ex.submit(worker, n): n for n in range(n_levels)}
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                r = fut.result()
                results[n] = r
                print("  n_sub = {}, N_faces = {}, N_vert = {}".format(
                    r['n_sub'], r['N_faces'], r['N_vert']), flush=True)
            except Exception as e:
                print("  n_sub = {} FAILED: {}".format(n, e), flush=True)

    print()
    print("  {:>6s}  {:>9s}  {:>9s}  {:>12s}  {:>12s}".format(
        "n_sub", "N_faces", "N_vert", "a0/(4pi)", "a2/(4pi)"))
    print("  " + "-" * 60)

    data = [results[n] for n in sorted(results)]
    for d in data:
        print("  {:>6d}  {:>9d}  {:>9d}  {:>12.6f}  {:>12.6f}".format(
            d['n_sub'], d['N_faces'], d['N_vert'], d['c0'], d['c1']))

    print()
    print("  Targets:  a0/(4pi) = 1.000000,  a2/(4pi) = 0.333333")
    print()

    # ── Richardson extrapolation ──
    print("=" * 92)
    print("  Richardson extrapolation  Q_rich = (4 Q(h/2) - Q(h)) / 3")
    print("=" * 92)
    print()
    print("  {:>9s}  {:>12s}  {:>15s}  {:>15s}  {:>15s}".format(
        "N_faces", "a2(h)", "a2 rich (1st)", "a2 rich (2nd)", "a2 rich (3rd)"))
    print("  " + "-" * 72)

    a2_vals = [d['c1'] for d in data]
    Nf_vals = [d['N_faces'] for d in data]

    # 1st-order Richardson
    a2_r1 = []
    for k in range(1, len(a2_vals)):
        r = (4 * a2_vals[k] - a2_vals[k - 1]) / 3
        a2_r1.append(r)

    # 2nd-order Richardson (need ratio 4 each level, and doubly-nested)
    a2_r2 = []
    for k in range(1, len(a2_r1)):
        r = (4 * a2_r1[k] - a2_r1[k - 1]) / 3
        a2_r2.append(r)

    # 3rd-order
    a2_r3 = []
    for k in range(1, len(a2_r2)):
        r = (4 * a2_r2[k] - a2_r2[k - 1]) / 3
        a2_r3.append(r)

    for i, d in enumerate(data):
        r1 = a2_r1[i - 1] if 0 < i <= len(a2_r1) else float('nan')
        r2 = a2_r2[i - 2] if 1 < i <= len(a2_r2) + 1 else float('nan')
        r3 = a2_r3[i - 3] if 2 < i <= len(a2_r3) + 2 else float('nan')
        print("  {:>9d}  {:>12.6f}  {:>15.6f}  {:>15.6f}  {:>15.6f}".format(
            d['N_faces'], d['c1'], r1, r2, r3))

    print()
    print("=" * 92)
    print("  Convergence to target 1/3 = 0.333333")
    print("=" * 92)
    print()
    if a2_r3:
        best = a2_r3[-1]
        dev = abs(best - 1/3) / (1/3) * 100
        print("  Best estimate (3rd Richardson): a2/(4 pi) = {:.6f}".format(best))
        print("  Deviation from 1/3: {:.4f}%".format(dev))
    if a2_r2:
        best2 = a2_r2[-1]
        dev2 = abs(best2 - 1/3) / (1/3) * 100
        print("  Best estimate (2nd Richardson): a2/(4 pi) = {:.6f}".format(best2))
        print("  Deviation from 1/3: {:.4f}%".format(dev2))
    print()


if __name__ == "__main__":
    main()