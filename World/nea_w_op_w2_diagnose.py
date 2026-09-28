#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_w_op_w2_diagnose.py

OP-W2 诊断版。

不拟合，直接打印：
  - 低本征值与 ℓ(ℓ+1) 的匹配
  - t*K(t) 在细密 t 网格上的完整形状
  - 最小二乘拟合的结果
  - 与连续精确公式的对比

连续精确公式（单位球面）：
  K_cont(t) = sum_{l=0}^{l_max} (2l+1) exp(-t l(l+1))
  t*K_cont(t) -> 1 + t/3 + ...

判决：
  如果离散 t*K(t) 在小 t 区域接近 1 + t/3，斜率就是 a2/(4pi) = 1/3
  如果 t*K(t) 明显偏离，说明网格本身有系统性偏差
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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
        A0 = (1.0/8.0) * (e01*c2 + e02*c1)
        A1 = (1.0/8.0) * (e01*c2 + e12*c0)
        A2 = (1.0/8.0) * (e02*c1 + e12*c0)
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


def K_continuous(t, l_max=50):
    """Exact K(t) for unit 2-sphere, truncated at l_max."""
    l = np.arange(0, l_max + 1)
    return np.sum((2*l + 1) * np.exp(-t * l * (l + 1)))


def main():
    print("=" * 92)
    print("  OP-W2 diagnosis: what does the discrete K(t) actually look like?")
    print("=" * 92)
    print()

    verts, faces = init_octahedron()
    for _ in range(4):
        verts, faces = subdivide(verts, faces)

    N = len(verts)
    print("  Mesh: N_vert = {}, N_faces = {}".format(N, len(faces)))
    print()

    L, M, diag_M = build_L_and_M(verts, faces)
    M_inv_sqrt = sp.diags(1.0 / np.sqrt(diag_M))
    L_sym = M_inv_sqrt @ L @ M_inv_sqrt
    L_sym = 0.5 * (L_sym + L_sym.T)

    print("  Full diagonalization (N = {})...".format(N), flush=True)
    evals = la.eigvalsh(L_sym.toarray())
    evals = np.sort(evals)
    evals_nz = evals[evals > 1e-10]

    print("  First 20 eigenvalues:")
    print("    " + " ".join("{:.4f}".format(x) for x in evals_nz[:20]))
    print()

    print("  Continuous l(l+1) reference:")
    l_arr = np.arange(1, 7)
    print("    " + " ".join("{:.4f}".format(l*(l+1)) for l in l_arr))
    print()

    lam_max = evals_nz[-1]
    lam_min_nz = evals_nz[0]
    print("  lambda_min_nz = {:.6f}".format(lam_min_nz))
    print("  lambda_max    = {:.4f}".format(lam_max))
    print("  1/lambda_max  = {:.6e}".format(1/lam_max))
    print("  1/lambda_min  = {:.6f}".format(1/lam_min_nz))
    print()

    # ── 直接打印 t*K(t) ──
    print("=" * 92)
    print("  t*K(t) on a fine t grid, compared with continuous formula")
    print("=" * 92)
    print()
    print("  {:>12s}  {:>14s}  {:>14s}  {:>14s}".format(
        "t", "t*K_disc", "t*K_cont", "diff"))
    print("  " + "-" * 60)

    t_grid = np.logspace(-4, 0, 25)
    for t in t_grid:
        K_d = np.sum(np.exp(-t * evals_nz))
        tK_d = t * K_d
        tK_c = t * K_continuous(t, 50)
        print("  {:>12.6f}  {:>14.6f}  {:>14.6f}  {:>+14.6f}".format(
            t, tK_d, tK_c, tK_d - tK_c))
    print()

    # ── 检查连续公式的渐近行为 ──
    print("=" * 92)
    print("  Continuous formula asymptote check")
    print("=" * 92)
    print()
    print("  Target:  t*K_cont(t) -> 1 + t/3 as t -> 0")
    print()
    print("  {:>12s}  {:>14s}  {:>14s}  {:>14s}".format(
        "t", "t*K_cont", "1 + t/3", "diff"))
    print("  " + "-" * 60)
    for t in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        tK_c = t * K_continuous(t, 100)
        asym = 1 + t/3
        print("  {:>12.6e}  {:>14.6f}  {:>14.6f}  {:>+14.6f}".format(
            t, tK_c, asym, tK_c - asym))
    print()

    # ── 多窗口拟合 ──
    print("=" * 92)
    print("  Multi-window fits: where does the slope 1/3 appear?")
    print("=" * 92)
    print()
    print("  {:>20s}  {:>12s}  {:>12s}".format(
        "t window", "intercept", "slope"))
    print("  " + "-" * 48)

    windows = [
        (1e-4, 1e-3),
        (1e-3, 1e-2),
        (1e-2, 5e-2),
        (5e-2, 1e-1),
        (1e-1, 3e-1),
        (3e-1, 1.0),
    ]
    for (t_lo, t_hi) in windows:
        t_arr = np.linspace(t_lo, t_hi, 30)
        tK = np.array([t * np.sum(np.exp(-t * evals_nz)) for t in t_arr])
        c1, c0 = np.polyfit(t_arr, tK, 1)
        print("  [{:.4e}, {:.4e}]  {:>12.6f}  {:>12.6f}".format(
            t_lo, t_hi, c0, c1))
    print()

    print("=" * 92)
    print("  Target: intercept -> 1.0, slope -> 1/3 = 0.333333")
    print("=" * 92)


if __name__ == "__main__":
    main()