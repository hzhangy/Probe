#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_unified_hessian_v2.py

Fixed zero-mode projection using scipy.linalg.orth.
Extended N range to verify s → 1.0 convergence.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def fibonacci_sphere(N):
    i = np.arange(N)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / max(1, N - 1)
    rho = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden * i
    dirs = np.column_stack([rho * np.cos(theta), y, rho * np.sin(theta)])
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def shell_positions(N, R=1.0, delta=0.05, seed=42):
    dirs = fibonacci_sphere(N)
    rng = np.random.default_rng(seed)
    radii = R + (rng.random(N) - 0.5) * delta
    return dirs * radii[:, None]


def yukawa_block(r_vec, r, Delta_w, m_w):
    r_hat = r_vec / r
    e = np.exp(-m_w * r)
    A = Delta_w * e / r**3
    cp = 1.0 + m_w * r
    cl = -(m_w**2 * r**2 + 3.0 * m_w * r + 3.0)
    return A * (cp * np.eye(3) + (cl - cp) * np.outer(r_hat, r_hat))


def gravity_block(r_vec, r, Delta_g):
    r_hat = r_vec / r
    return Delta_g * (np.eye(3) - 3.0 * np.outer(r_hat, r_hat)) / r**3


def build_unified_hessian(positions, Delta_g=0.134, Delta_em=0.134,
                           Delta_w=0.0, m_w=0.0, r_min=1e-12):
    N = positions.shape[0]
    H = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < r_min:
                continue
            blk = gravity_block(r_vec, r, Delta_g)
            blk += gravity_block(r_vec, r, Delta_em)
            if Delta_w > 0 and m_w > 0:
                blk += yukawa_block(r_vec, r, Delta_w, m_w)
            ii = slice(3 * i, 3 * i + 3)
            jj = slice(3 * j, 3 * j + 3)
            H[ii, jj] -= blk
            H[jj, ii] -= blk.T
            H[ii, ii] += blk
            H[jj, jj] += blk
    return 0.5 * (H + H.T)


def project_zero_modes(H, positions, tol=1e-10):
    """Use scipy.linalg.orth for stable orthonormal basis."""
    N = positions.shape[0]
    Z = np.zeros((3 * N, 6))
    for a in range(3):
        for i in range(N):
            Z[3 * i + a, a] = 1.0
    axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    for a_idx, ax in enumerate(axes):
        for i in range(N):
            dp = np.cross(ax, positions[i])
            for b in range(3):
                Z[3 * i + b, 3 + a_idx] = dp[b]

    # scipy.linalg.orth uses SVD with tolerance for numerical rank
    Q = la.orth(Z, rcond=tol)
    # Ensure Q is (3N, rank) with orthonormal columns
    if Q.shape[1] > 6:
        Q = Q[:, :6]
    P = np.eye(3 * N) - Q @ Q.T
    H_proj = P @ H @ P
    return 0.5 * (H_proj + H_proj.T)


def weyl_slope(evals, fit_lo, fit_hi):
    ev = np.sort(np.abs(evals))
    ev = ev[ev > 1e-10 * np.max(ev)]
    n = len(ev)
    if n < 20:
        return float("nan")
    i0, i1 = int(fit_lo * n), int(fit_hi * n)
    x = np.log(ev[i0:i1])
    y = np.log(np.arange(1, n + 1)[i0:i1])
    s, _ = np.polyfit(x, y, 1)
    return float(s)


def main():
    print("=" * 100)
    print("  UNIFIED Hessian v2  —  weak-dominant regime with stable projection")
    print("=" * 100)
    print()
    print("  Scan N ∈ {48, 96, 192, 384, 768}  with Δ_w=100, m_W=5, δ=0.05")
    print()

    Ns = [48, 96, 192, 384, 768]
    print("─" * 100)
    print(f"  {'N':>6s}  {'s[.1-.6]':>10s}  {'s[.3-.7]':>10s}"
          f"  {'s[.5-.9]':>10s}  {'d_spec=2s':>12s}")
    print("─" * 100)

    for N in Ns:
        pos = shell_positions(N, delta=0.05)
        H = build_unified_hessian(pos, Delta_w=100.0, m_w=5.0)
        H = project_zero_modes(H, pos)
        evals = la.eigvalsh(H)

        s_full = weyl_slope(evals, 0.10, 0.60)
        s_mid = weyl_slope(evals, 0.30, 0.70)
        s_hi = weyl_slope(evals, 0.50, 0.90)

        print(f"  {N:>6d}  {s_full:>10.4f}  {s_mid:>10.4f}"
              f"  {s_hi:>10.4f}  {2*s_full:>12.4f}")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()