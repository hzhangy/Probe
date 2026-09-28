#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_mass_tensor_product_audit.py

Tensor product of external tangent Hessian and internal NNI mass matrix:

    H_total = H_tangent ⊗ I_3 + I_{2N} ⊗ M²

Heat kernel factorizes:
    Tr(exp(-t H_total)) = Tr(exp(-t H_tangent)) · Tr(exp(-t M²))

Extract:
    Tr(M²)  →  electroweak VEV / Higgs mass term
    Tr(M⁴)  →  Higgs quartic coupling λ
"""

import numpy as np
import scipy.linalg as la


# ─────────────────────────────────────────────────────────────────────
# 1. M论文 NNI mass matrix
# ─────────────────────────────────────────────────────────────────────
def build_M_matrix():
    """
    M = m_e · [[1,  a, 0],
               [a,  A, d],
               [0,  d, B]]
    with a = 3ε = 0.3,  d = |2O| = 48,
         A = 66π,  B = 66π · 16π/3.
    """
    m_e = 0.51099895  # MeV
    a = 0.3
    A = 66.0 * np.pi
    B = 66.0 * np.pi * 16.0 * np.pi / 3.0
    d = 48.0

    M = m_e * np.array([
        [1.0, a, 0.0],
        [a, A, d],
        [0.0, d, B],
    ])
    return M


# ─────────────────────────────────────────────────────────────────────
# 2. Trace invariants
# ─────────────────────────────────────────────────────────────────────
def trace_invariants(M):
    M2 = M @ M
    M4 = M2 @ M2
    return np.trace(M), np.trace(M2), np.trace(M4)


# ─────────────────────────────────────────────────────────────────────
# 3. Tangent-space Hessian (small N for tensor-product test)
# ─────────────────────────────────────────────────────────────────────
def fibonacci_sphere(N):
    i = np.arange(N)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / max(1, N - 1)
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden * i
    return np.column_stack([r * np.cos(theta), y, r * np.sin(theta)])


def local_tangent_basis(p):
    p = p / np.linalg.norm(p)
    idx = int(np.argmin(np.abs(p)))
    ref = np.zeros(3)
    ref[idx] = 1.0
    e1 = ref - np.dot(ref, p) * p
    n1 = np.linalg.norm(e1)
    if n1 < 1e-8:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    e1 /= n1
    e2 = np.cross(p, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def build_tangent_hessian(positions, Delta=0.133975):
    N = positions.shape[0]
    D = Delta / (4.0 * np.pi)
    H = np.zeros((2 * N, 2 * N))
    T = np.zeros((N, 3, 2))
    for i, p in enumerate(positions):
        e1, e2 = local_tangent_basis(p)
        T[i, :, 0] = e1
        T[i, :, 1] = e2
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 0.01:
                continue
            r_hat = r_vec / r
            blk3 = (np.eye(3) - 3.0 * np.outer(r_hat, r_hat)) / r ** 3
            blk3 *= D
            blk2 = T[i].T @ blk3 @ T[j]
            ii = slice(2 * i, 2 * i + 2)
            jj = slice(2 * j, 2 * j + 2)
            H[ii, jj] += -blk2
            H[jj, ii] += -blk2
            H[ii, ii] += blk2
            H[jj, jj] += blk2
    H = 0.5 * (H + H.T)
    return H


def remove_rotation_zero_modes(H, positions):
    N = positions.shape[0]
    axes = [np.array([1., 0., 0.]),
            np.array([0., 1., 0.]),
            np.array([0., 0., 1.])]
    R = np.zeros((2 * N, 3))
    for a_idx, ax in enumerate(axes):
        for i in range(N):
            p = positions[i]
            e1, e2 = local_tangent_basis(p)
            dp = np.cross(ax, p)
            R[2 * i, a_idx] = np.dot(dp, e1)
            R[2 * i + 1, a_idx] = np.dot(dp, e2)
    Q, _ = np.linalg.qr(R)
    P = np.eye(2 * N) - Q @ Q.T
    H_proj = P @ H @ P
    H_proj = 0.5 * (H_proj + H_proj.T)
    return H_proj


# ─────────────────────────────────────────────────────────────────────
# 4. Tensor-product heat kernel
# ─────────────────────────────────────────────────────────────────────
def heat_kernel(evals, t):
    return np.exp(-t * evals).sum()


def main():
    print("=" * 86)
    print("  Mass matrix × tangent Hessian: trace invariants and heat kernel")
    print("=" * 86)
    print()

    # ── M matrix ──
    M = build_M_matrix()
    evals_M = np.linalg.eigvalsh(M)

    print("[1] M matrix (NNI, M paper):")
    print(f"    eigenvalues (MeV): {evals_M}")
    print(f"    ratios (m_e = 1):  {evals_M / evals_M[0]}")
    print()

    Tr_M, Tr_M2, Tr_M4 = trace_invariants(M)
    print("[2] Trace invariants:")
    print(f"    Tr(M)   = {Tr_M:.6e}  MeV")
    print(f"    Tr(M²)  = {Tr_M2:.6e}  MeV²")
    print(f"    Tr(M⁴)  = {Tr_M4:.6e}  MeV⁴")
    print()

    # ── Dimensionless ratios ──
    r1 = Tr_M4 / Tr_M2 ** 2
    r2 = Tr_M2 / Tr_M ** 2
    r3 = Tr_M2 / Tr_M

    print("[3] Dimensionless ratios:")
    print(f"    Tr(M⁴) / Tr(M²)²  = {r1:.6f}")
    print(f"    Tr(M²) / Tr(M)²   = {r2:.6f}")
    print(f"    Tr(M²) / Tr(M)    = {r3:.6f}")
    print()
    print(f"    Higgs quartic target: λ = 1/8 = 0.125")
    print(f"    Ratio Tr(M⁴)/Tr(M²)² = {r1:.6f}")
    print(f"    8 × Ratio = {8*r1:.6f}")
    print()

    # ── Normalized Yukawa matrix ──
    v_h_MeV = 245.604e3
    Y = M / v_h_MeV
    Tr_Y2 = np.trace(Y @ Y)
    Tr_Y4 = np.trace(Y @ Y @ Y @ Y)
    print("[4] Yukawa matrix Y = M / v_h:")
    print(f"    v_h = {v_h_MeV:.1f} MeV")
    print(f"    Tr(Y²) = {Tr_Y2:.6e}")
    print(f"    Tr(Y⁴) = {Tr_Y4:.6e}")
    print(f"    Tr(Y⁴) / Tr(Y²)² = {Tr_Y4 / Tr_Y2**2:.6f}")
    print(f"    Expected λ = 1/8 = 0.125")
    print(f"    Match: {Tr_Y4 / Tr_Y2**2 / 0.125 * 100:.4f}% of 1/8")
    print()

    # ── Tensor product with tangent Hessian ──
    print("[5] Tensor product H_tangent ⊗ I_3 + I_{2N} ⊗ M²:")
    N = 48
    pos = fibonacci_sphere(N)
    H_tan = build_tangent_hessian(pos)
    H_tan = remove_rotation_zero_modes(H_tan, pos)

    evals_tan = la.eigvalsh(H_tan)
    evals_M2 = evals_M ** 2

    # Tensor product eigenvalues (in appropriate unit scaling)
    # We work in dimensionless units: scale M by a reference mass
    # Use m_e × v_h normalization
    scale = (evals_M[1] / evals_M[0])  # use μ/e ratio as dimensionless scale
    # Actually, let's just do direct tensor product
    H_total = np.kron(H_tan, np.eye(3)) + np.kron(np.eye(2*N), np.eye(3) * 0)
    # Add M² properly with energy scale matching
    # H_total = H_tangent ⊗ I_3 + I_{2N} ⊗ M²
    # But M is in MeV, H_tangent in dimensionless - need scaling
    # Let's use H_tangent in "units of Δ/4π" and M in "units of v_h"
    M_dimless = M / v_h_MeV
    H_total = np.kron(H_tan, np.eye(3)) + np.kron(np.eye(2*N), M_dimless @ M_dimless)

    evals_total = la.eigvalsh(H_total)
    print(f"    dim(H_total) = {H_total.shape[0]}")
    print(f"    min eigenvalue: {evals_total[0]:.6e}")
    print(f"    max eigenvalue: {evals_total[-1]:.6e}")
    print()

    # Verify heat kernel factorization
    print("[6] Heat kernel factorization check:")
    print(f"    {'t':>10s}  {'Tr(e^-tH_total)':>18s}  "
          f"{'Tr(e^-tH_tan)·Tr(e^-tM²)':>26s}  {'ratio':>10s}")
    print("  " + "-" * 70)
    for t in [0.01, 0.05, 0.1, 0.5, 1.0]:
        K_total = np.sum(np.exp(-t * evals_total))
        K_tan = np.sum(np.exp(-t * evals_tan))
        K_M = np.sum(np.exp(-t * evals_M2 / v_h_MeV**2))
        K_product = K_tan * K_M
        ratio = K_total / K_product if K_product > 0 else float("nan")
        print(f"    {t:10.4f}  {K_total:18.6e}  {K_product:26.6e}  {ratio:10.6f}")


if __name__ == "__main__":
    main()