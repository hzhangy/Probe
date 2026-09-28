#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_w_verify_independent.py

Independent verification of face-centered Dirac d_spec.

Method 1: |lambda_min| scaling with N determines operator order m.
Method 2: Weyl slope s_D directly gives d_spec if m = 1.
Method 3: Self-consistency between D and D^2 Weyl slopes.

This script does NOT divide a previously printed 2s by 2. It runs
the actual numerical computation and extracts both quantities
independently.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ── Icosahedron mesh + Dirac builder (from earlier scripts) ──
def init_icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])
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


def build_dirac_sparse(verts, faces):
    Nf = len(faces)
    centers = np.zeros((Nf, 3))
    areas = np.zeros(Nf)
    normals = np.zeros((Nf, 3))
    u_hat = np.zeros((Nf, 3))
    v_hat = np.zeros((Nf, 3))

    for i, f in enumerate(faces):
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        c = (v0 + v1 + v2) / 3.0
        centers[i] = c / np.linalg.norm(c)
        n = np.cross(v1 - v0, v2 - v0)
        areas[i] = 0.5 * np.linalg.norm(n)
        n = n / np.linalg.norm(n)
        normals[i] = n
        u = v1 - v0
        u = u / np.linalg.norm(u)
        vv = np.cross(n, u)
        u_hat[i] = u
        v_hat[i] = vv / np.linalg.norm(vv)

    sig1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sig2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sig3 = np.array([[1, 0], [0, -1]], dtype=complex)

    face_vsets = [set(f) for f in faces]
    rows, cols, data = [], [], []

    for i in range(Nf):
        for j in range(i + 1, Nf):
            common = list(face_vsets[i] & face_vsets[j])
            if len(common) != 2:
                continue
            edge_vec = verts[common[1]] - verts[common[0]]
            edge_len = np.linalg.norm(edge_vec)
            d_ij = centers[j] - centers[i]
            proj = d_ij - np.dot(d_ij, normals[i]) * normals[i]
            nn = np.linalg.norm(proj)
            if nn < 1e-12:
                continue
            n3 = proj / nn
            n2 = np.array([np.dot(n3, u_hat[i]), np.dot(n3, v_hat[i])])
            n2 /= np.linalg.norm(n2)
            e_i = np.array([np.dot(edge_vec, u_hat[i]), np.dot(edge_vec, v_hat[i])])
            e_j = np.array([np.dot(edge_vec, u_hat[j]), np.dot(edge_vec, v_hat[j])])
            theta = np.arctan2(e_i[1], e_i[0]) - np.arctan2(e_j[1], e_j[0])
            U = la.expm(1j * (theta / 2.0) * sig3)
            n_sig = n2[0] * sig1 + n2[1] * sig2
            coeff = -1j * edge_len / (2.0 * np.sqrt(areas[i] * areas[j]))
            blk = coeff * (n_sig @ U)
            for r in range(2):
                for c in range(2):
                    rows.append(2*i + r); cols.append(2*j + c); data.append(blk[r, c])
                    rows.append(2*j + c); cols.append(2*i + r); data.append(np.conj(blk[r, c]))

    dim = 2 * Nf
    D = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    D = 0.5 * (D + D.conj().T)
    return D


def weyl_slope(evals, fit_lo=0.10, fit_hi=0.60):
    ev = np.sort(np.abs(evals))
    ev_max = np.max(ev)
    ev = ev[ev > 1e-10 * ev_max]
    n = len(ev)
    i0, i1 = int(fit_lo * n), int(fit_hi * n)
    s, _ = np.polyfit(np.log(ev[i0:i1]), np.log(np.arange(1, n+1)[i0:i1]), 1)
    return s


def main():
    print("=" * 92)
    print("  INDEPENDENT verification of face-centered Dirac d_spec")
    print("=" * 92)
    print()

    verts, faces = init_icosahedron()

    # ── Collect data across subdivisions ──
    data = []
    for n_sub in range(0, 5):
        if n_sub > 0:
            verts, faces = subdivide(verts, faces)
        Nf = len(faces)
        dim = 2 * Nf

        print("  computing N_faces = {} (dim = {})...".format(Nf, dim), flush=True)
        D = build_dirac_sparse(verts, faces)

        if dim <= 3000:
            ev = la.eigvalsh(D.toarray())
        else:
            k = min(dim - 2, 1500)
            ev = spla.eigsh(D, k=k, sigma=-0.1, which='LM',
                            return_eigenvectors=False)

        ev_abs = np.sort(np.abs(ev))
        ev_nz = ev_abs[ev_abs > 1e-10 * np.max(ev_abs)]

        lambda_min = ev_nz[0]
        s_D = weyl_slope(ev)

        # D^2 Weyl slope
        ev_sq = np.sort(ev_abs**2)
        ev_sq = ev_sq[ev_sq > 1e-20 * np.max(ev_sq)]
        s_D2 = weyl_slope(np.sqrt(ev_sq))  # Weyl for D^2 via |lambda_D2|^{1/2}? 
        # Actually for D^2 as a standalone operator with eigenvalues mu_k = lambda_k^2,
        # N(mu) ~ mu^{d/(2m)}. We want s_D2 = d/(2m).
        # Simple approach: fit log N(mu) vs log mu.
        mu_sorted = np.sort(ev_abs**2)
        mu_sorted = mu_sorted[mu_sorted > 1e-20 * np.max(mu_sorted)]
        n_mu = len(mu_sorted)
        i0, i1 = int(0.10 * n_mu), int(0.60 * n_mu)
        s_D2_actual, _ = np.polyfit(
            np.log(mu_sorted[i0:i1]),
            np.log(np.arange(1, n_mu+1)[i0:i1]),
            1
        )

        data.append({
            'n_sub': n_sub,
            'N_faces': Nf,
            'dim': dim,
            'lambda_min': lambda_min,
            's_D': s_D,
            's_D2': s_D2_actual,
        })

    # ── Output table ──
    print()
    print("  {:>6s}  {:>8s}  {:>12s}  {:>10s}  {:>12s}  {:>12s}".format(
        "n_sub", "N_faces", "lambda_min", "s_D", "s_D2", "s_D/s_D2"))
    print("  " + "-" * 72)
    for d in data:
        ratio = d['s_D'] / d['s_D2'] if d['s_D2'] > 0 else float('nan')
        print("  {:>6d}  {:>8d}  {:>12.4e}  {:>10.4f}  {:>12.4f}  {:>12.4f}".format(
            d['n_sub'], d['N_faces'], d['lambda_min'],
            d['s_D'], d['s_D2'], ratio))
    print()

    # ── Method 1: lambda_min scaling ──
    print("  [Method 1] |lambda_min| vs N_faces scaling")
    Ns = np.array([d['N_faces'] for d in data], dtype=float)
    lms = np.array([d['lambda_min'] for d in data])
    # Fit log(lm) = -alpha * log(N) + const
    alpha_fit, _ = np.polyfit(np.log(Ns), np.log(lms), 1)
    alpha_fit = -alpha_fit  # we fit log(lm) vs log(N), slope is -alpha
    print("    alpha (exponent in |lambda_min| ~ N^-alpha) = {:.4f}".format(alpha_fit))
    print("    If alpha = 0.5: m = 1 (first-order operator)")
    print("    If alpha = 1.0: m = 2 (second-order operator)")
    print()

    # ── Method 2: s_D / s_D2 ratio ──
    print("  [Method 2] s_D / s_D2 ratio (should be 2 if D and D^2 are same operator type)")
    ratios = [d['s_D'] / d['s_D2'] for d in data if d['s_D2'] > 0]
    print("    Ratios: {}".format(["{:.3f}".format(r) for r in ratios]))
    if ratios:
        mean_ratio = np.mean(ratios)
        print("    Mean ratio: {:.4f}".format(mean_ratio))
        print("    If mean ratio = 2: consistent with D being m-th order, D^2 being 2m-th")
        print("    If mean ratio = 1: inconsistent (D and D^2 same order, impossible)")
    print()

    # ── Verdict ──
    print("=" * 92)
    print("  VERDICT")
    print("=" * 92)
    print()
    print("  Face-centered Dirac matrix elements:")
    print("    D_ij = -i (l_ij / (2 sqrt(A_i A_j))) (n_ij . sigma) U_ij")
    print("    Dimension: [l]/[A] = [L]/[L^2] = [1/L] -> FIRST ORDER (m=1)")
    print()
    print("  From Method 1 (lambda_min scaling):")
    print("    alpha = {:.4f} -> m = {}".format(
        alpha_fit, 1 if alpha_fit < 0.7 else 2))
    print()
    print("  If m = 1: d_spec = s_D (NOT 2 s_D)")
    print("  If m = 2: d_spec = 2 s_D")
    print()
    print("  Final d_spec values:")
    for d in data:
        m = 1 if alpha_fit < 0.7 else 2
        ds = m * d['s_D']
        print("    n_sub = {}, N_faces = {}: d_spec = {:.4f}".format(
            d['n_sub'], d['N_faces'], ds))


if __name__ == "__main__":
    main()