#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_four_force_v2.py

Four-force model with STRONG as LINEAR CONFINEMENT potential.

Physical motivation:
  Strong force in N.E.A. is K4 cycle locking → color confinement
  → linear potential V_s(r) = -σ r  (Cornell-like)

This is NOT Yukawa. Its Hessian has a different Fourier structure:
  - 1/r  : k^{-2} kernel
  - Yukawa: flat kernel at large k
  - linear: k^{-4} kernel (more singular)
"""

import numpy as np
import scipy.linalg as la


def fibonacci_sphere(N):
    i = np.arange(N)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / max(1, N - 1)
    rho = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden * i
    dirs = np.column_stack([rho * np.cos(theta), y, rho * np.sin(theta)])
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def gaussian_shell(N, R=1.0, sigma=0.05, seed=42):
    dirs = fibonacci_sphere(N)
    rng = np.random.default_rng(seed)
    radii = R + sigma * rng.standard_normal(N)
    return dirs * radii[:, None]


def yukawa_block(r_vec, r, D_w, m_w):
    r_hat = r_vec / r
    e = np.exp(-m_w * r)
    A = D_w * e / r**3
    cp = 1.0 + m_w * r
    cl = -(m_w**2 * r**2 + 3.0 * m_w * r + 3.0)
    return A * (cp * np.eye(3) + (cl - cp) * np.outer(r_hat, r_hat))


def gravity_block(r_vec, r, D_g):
    r_hat = r_vec / r
    return D_g * (np.eye(3) - 3.0 * np.outer(r_hat, r_hat)) / r**3


def linear_conf_block(r_vec, r, D_s):
    """
    Hessian of V(r) = -D_s r (linear confinement).

    ∂²V/∂x_a∂x_b = -D_s (δ_ab - r̂_a r̂_b) / r
    """
    r_hat = r_vec / r
    return -D_s * (np.eye(3) - np.outer(r_hat, r_hat)) / r


def log_conf_block(r_vec, r, D_s):
    """
    Hessian of V(r) = -D_s ln(r) (2D logarithmic).

    ∂²V/∂x_a∂x_b = D_s (δ_ab - 2 r̂_a r̂_b) / r²
    """
    r_hat = r_vec / r
    return D_s * (np.eye(3) - 2.0 * np.outer(r_hat, r_hat)) / r**2


def build_hessian(positions, D_g=0.134, D_em=0.134,
                   D_w=100.0, m_w=5.0,
                   D_s=0.0, m_s=5.0, strong_type='yukawa',
                   r_min=1e-12):
    N = positions.shape[0]
    H = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < r_min:
                continue
            blk = gravity_block(r_vec, r, D_g) + gravity_block(r_vec, r, D_em)
            if D_w > 0 and m_w > 0:
                blk += yukawa_block(r_vec, r, D_w, m_w)
            if D_s > 0:
                if strong_type == 'yukawa':
                    blk += yukawa_block(r_vec, r, D_s, m_s)
                elif strong_type == 'linear':
                    blk += linear_conf_block(r_vec, r, D_s)
                elif strong_type == 'log':
                    blk += log_conf_block(r_vec, r, D_s)
            ii = slice(3 * i, 3 * i + 3)
            jj = slice(3 * j, 3 * j + 3)
            H[ii, jj] -= blk
            H[jj, ii] -= blk.T
            H[ii, ii] += blk
            H[jj, jj] += blk
    return 0.5 * (H + H.T)


def project_zero_modes(H, positions, tol=1e-10):
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
    Q = la.orth(Z, rcond=tol)
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
    print("  Four-force Hessian: weak Yukawa + STRONG (linear/log/Yukawa)")
    print("=" * 100)
    print()

    Ns = [48, 96, 192, 384]
    sigma = 0.05

    # ============================================================
    # SCAN 1: strong-force TYPE
    # ============================================================
    print("─" * 100)
    print("  SCAN 1: strong-force type dependence (D_s=100, δ=0.05)")
    print("─" * 100)
    print(f"  {'type':>12s}  {'N=48':>10s}  {'N=96':>10s}  {'N=192':>10s}"
          f"  {'N=384':>10s}  {'2s(N=384)':>12s}")
    print("─" * 100)
    for stype in ['none', 'yukawa', 'linear', 'log']:
        line = f"  {stype:>12s}"
        s_384 = None
        for N in Ns:
            pos = gaussian_shell(N, sigma=sigma)
            Ds = 0.0 if stype == 'none' else 100.0
            H = build_hessian(pos, D_w=100.0, m_w=5.0,
                               D_s=Ds, m_s=5.0, strong_type=stype if stype != 'none' else 'yukawa')
            H = project_zero_modes(H, pos)
            evals = la.eigvalsh(H)
            s = weyl_slope(evals, 0.10, 0.60)
            line += f"  {s:>10.4f}"
            if N == 384:
                s_384 = s
        line += f"  {2*s_384:>12.4f}"
        print(line)
    print()

    # ============================================================
    # SCAN 2: linear D_s dependence
    # ============================================================
    print("─" * 100)
    print("  SCAN 2: linear confinement D_s dependence (D_w=100, m_w=5, δ=0.05)")
    print("─" * 100)
    print(f"  {'D_s':>8s}  {'N=48':>10s}  {'N=96':>10s}  {'N=192':>10s}"
          f"  {'N=384':>10s}  {'2s(N=384)':>12s}")
    print("─" * 100)
    for D_s in [0.0, 0.1, 1.0, 10.0, 100.0]:
        line = f"  {D_s:>8.2f}"
        s_384 = None
        for N in Ns:
            pos = gaussian_shell(N, sigma=sigma)
            H = build_hessian(pos, D_w=100.0, m_w=5.0,
                               D_s=D_s, strong_type='linear')
            H = project_zero_modes(H, pos)
            evals = la.eigvalsh(H)
            s = weyl_slope(evals, 0.10, 0.60)
            line += f"  {s:>10.4f}"
            if N == 384:
                s_384 = s
        line += f"  {2*s_384:>12.4f}"
        print(line)
    print()

    # ============================================================
    # SCAN 3: weak + linear strong (both dominated)
    # ============================================================
    print("─" * 100)
    print("  SCAN 3: WEAK + LINEAR STRONG (D_w=D_s=100, δ=0.05)")
    print("─" * 100)
    print(f"  {'N':>6s}  {'s[.1-.3]':>10s}  {'s[.3-.5]':>10s}"
          f"  {'s[.5-.7]':>10s}  {'s[.7-.9]':>10s}"
          f"  {'2s[.3-.7]':>12s}")
    print("─" * 100)
    for N in Ns:
        pos = gaussian_shell(N, sigma=sigma)
        H = build_hessian(pos, D_w=100.0, m_w=5.0,
                           D_s=100.0, strong_type='linear')
        H = project_zero_modes(H, pos)
        evals = la.eigvalsh(H)
        s1 = weyl_slope(evals, 0.10, 0.30)
        s2 = weyl_slope(evals, 0.30, 0.50)
        s3 = weyl_slope(evals, 0.50, 0.70)
        s4 = weyl_slope(evals, 0.70, 0.90)
        s_mid = weyl_slope(evals, 0.30, 0.70)
        print(f"  {N:>6d}  {s1:>10.4f}  {s2:>10.4f}"
              f"  {s3:>10.4f}  {s4:>10.4f}  {2*s_mid:>12.4f}")
    print()

    print("=" * 100)
    print("  Interpretation:")
    print("    - Yukawa strong + Yukawa weak  : linear superposition, no change")
    print("    - LINEAR strong (confinement)   : different kernel, may shift s")
    print("    - LOG strong (2D)               : δ-like Hessian, s → 0")
    print("=" * 100)


if __name__ == "__main__":
    main()