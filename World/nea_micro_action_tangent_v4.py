#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_micro_action_tangent_v4.py

CORRECTED version of nea_micro_action_tangent_v2/v3.

Critical fix in build_tangent_hessian (v2/v3 bug):
  - Diagonal block H_ii must use T_i^T (blk3_ij) T_i  (only i's basis)
  - v2/v3 incorrectly used T_i^T (blk3_ij) T_j  in the diagonal block.
  - Off-diagonal H_ji must be H_ij^T (v2/v3 did not transpose).

Mathematical derivation (V_ij = D / r_ij, r_ij = |r_i - r_j|):

  ∂²V_ij / ∂r_i^a ∂r_i^b = -D (δ^ab - 3 r̂^a r̂^b) / r³   ≡ -blk3
  ∂²V_ij / ∂r_i^a ∂r_j^b = +D (δ^ab - 3 r̂^a r̂^b) / r³   ≡ +blk3

Projected onto the tangent space of S²:
  H_ii = Σ_{j≠i} T_i^T (-blk3_ij) T_i
  H_ij = T_i^T (+blk3_ij) T_j
  H_ji = H_ij^T

Action:  S = - Σ_{i<j} Delta / (4 π r_ij)
"""

import numpy as np
import scipy.linalg as la
from scipy.spatial.distance import pdist


def fibonacci_sphere(N):
    """N quasi-uniform points on S²."""
    i = np.arange(N)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / max(1, N - 1)
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden * i
    pts = np.column_stack([r * np.cos(theta), y, r * np.sin(theta)])
    # Robust renormalization
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def local_tangent_basis(p):
    """Orthonormal tangent basis (e1, e2) at p ∈ S²."""
    p = p / np.linalg.norm(p)
    idx = int(np.argmin(np.abs(p)))
    ref = np.zeros(3)
    ref[idx] = 1.0
    e1 = ref - np.dot(ref, p) * p
    n1 = np.linalg.norm(e1)
    if n1 < 1e-12:
        trial = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(p, trial)
        if np.linalg.norm(e1) < 1e-12:
            trial = np.array([0.0, 1.0, 0.0])
            e1 = np.cross(p, trial)
        e1 = e1 / np.linalg.norm(e1)
    else:
        e1 = e1 / n1
    e2 = np.cross(p, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def build_tangent_hessian(positions, Delta=0.133975, r_min=1e-10):
    """
    Tangent-space Hessian of S = - Σ_{i<j} Delta / (4 π r_ij).

    For each pair (i, j), i < j:
        blk3_ij = D (I - 3 r̂ r̂ᵀ) / r³,  D = Delta/(4π)
        H_ij   += T_iᵀ (blk3_ij) T_j
        H_ji   += (H_ij)ᵀ
        H_ii   += - T_iᵀ (blk3_ij) T_i
        H_jj   += - T_jᵀ (blk3_ij) T_j
    """
    N = positions.shape[0]
    D = Delta / (4.0 * np.pi)
    H = np.zeros((2 * N, 2 * N))

    # Precompute tangent bases
    T = np.zeros((N, 3, 2))
    for i, p in enumerate(positions):
        e1, e2 = local_tangent_basis(p)
        T[i, :, 0] = e1
        T[i, :, 1] = e2

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < r_min:
                continue
            r_hat = r_vec / r
            blk3 = D * (np.eye(3) - 3.0 * np.outer(r_hat, r_hat)) / r ** 3

            ii = slice(2 * i, 2 * i + 2)
            jj = slice(2 * j, 2 * j + 2)

            # Off-diagonal block (2 × 2)
            H_ij = T[i].T @ blk3 @ T[j]

            # Diagonal contributions (2 × 2)
            H_ii = -T[i].T @ blk3 @ T[i]
            H_jj = -T[j].T @ blk3 @ T[j]

            H[ii, jj] += H_ij
            H[jj, ii] += H_ij.T          # enforce H_ji = H_ijᵀ
            H[ii, ii] += H_ii
            H[jj, jj] += H_jj

    # Symmetrization (removes floating-point roundoff)
    return 0.5 * (H + H.T)


def remove_rotation_zero_modes(H, positions, tol=1e-8):
    """
    Project out the 3 SO(3) rotation zero modes.

    Rotation vectors δr_i = ω_a × r_i for a ∈ {x, y, z}.
    In tangent coordinates:
        R[2i,   a] = (ω_a × r_i) · e1_i
        R[2i+1, a] = (ω_a × r_i) · e2_i

    Uses SVD-truncated orthonormal basis (numerically robust).
    """
    N = positions.shape[0]
    axes = [np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])]
    R = np.zeros((2 * N, 3))
    for a_idx, ax in enumerate(axes):
        for i in range(N):
            p = positions[i]
            e1, e2 = local_tangent_basis(p)
            dp = np.cross(ax, p)
            R[2 * i, a_idx] = np.dot(dp, e1)
            R[2 * i + 1, a_idx] = np.dot(dp, e2)

    U, s, _ = np.linalg.svd(R, full_matrices=False)
    if s[0] < 1e-14:
        # Degenerate: no rotation modes
        return H
    rank = int(np.sum(s > tol * s[0]))
    Q = U[:, :rank]
    P = np.eye(2 * N) - Q @ Q.T

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        H_proj = P @ H @ P
    return 0.5 * (H_proj + H_proj.T)


def weyl_slope_fixed(evals, fit_lo=0.10, fit_hi=0.60):
    """Fixed-window Weyl fit of |λ| (same window for all N)."""
    ev = np.sort(np.abs(evals))
    ev = ev[ev > 1e-10 * np.max(ev)]
    n = len(ev)
    if n < 40:
        return float("nan")
    i0, i1 = int(fit_lo * n), int(fit_hi * n)
    s, _ = np.polyfit(np.log(ev[i0:i1]),
                      np.log(np.arange(1, n + 1)[i0:i1]), 1)
    return float(s)


def weyl_slope_adaptive(evals, min_points=30):
    """Adaptive-window Weyl fit (diagnostic)."""
    ev = np.sort(evals)
    ev = ev[ev > 1e-8 * np.max(np.abs(ev))]
    n = len(ev)
    if n < min_points:
        return float("nan"), None
    log_ev = np.log(ev)
    log_counts = np.log(np.arange(1, n + 1))
    best = (-np.inf, float("nan"), None)
    for f_lo in np.linspace(0.10, 0.40, 7):
        for f_hi in np.linspace(0.60, 0.90, 7):
            i0, i1 = int(f_lo * n), int(f_hi * n)
            if i1 - i0 < min_points:
                continue
            x, y = log_ev[i0:i1], log_counts[i0:i1]
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r2 > best[0]:
                best = (r2, slope, (f_lo, f_hi, r2))
    return float(best[1]), best[2]


def main():
    print("=" * 92)
    print("  Micro-action Hessian (v4 CORRECTED): SVD projection, fixed window")
    print("=" * 92)
    print()
    print("  Action:  S = - Σ_{i<j} Delta / (4 π r_ij)")
    print("  Tangent space: 2N dim;  SO(3) zero modes projected out (SVD)")
    print()

    Ns = [48, 96, 192, 384, 768, 1536]

    print("─" * 92)
    print(f"  {'N':>6s}  {'dim':>6s}  {'r_min':>10s}  {'r_mean':>10s}"
          f"  {'d_spec(fix)':>12s}  {'R^2(fix)':>10s}"
          f"  {'d_spec(adap)':>13s}  {'window':>12s}")
    print("─" * 92)

    results = []
    for N in Ns:
        pos = fibonacci_sphere(N)

        # Geometry diagnostics (fast via pdist)
        dists = pdist(pos)
        r_min_actual = float(dists.min())
        r_mean_actual = float(dists.mean())

        H = build_tangent_hessian(pos)
        H = remove_rotation_zero_modes(H, pos)
        evals = la.eigvalsh(H)

        # Fixed-window (main result)
        s_fix = weyl_slope_fixed(evals, 0.10, 0.60)
        d_fix = 2.0 * s_fix if not np.isnan(s_fix) else float("nan")

        # R² for the fixed window
        ev = np.sort(np.abs(evals))
        ev = ev[ev > 1e-10 * np.max(ev)]
        n = len(ev)
        i0, i1 = int(0.10 * n), int(0.60 * n)
        x = np.log(ev[i0:i1])
        y = np.log(np.arange(1, n + 1)[i0:i1])
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        y_pred = slope * x + intercept
        r2_fix = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

        # Adaptive-window (diagnostic only)
        s_adap, window = weyl_slope_adaptive(evals)
        d_adap = 2.0 * s_adap if not np.isnan(s_adap) else float("nan")
        win_str = f"[{window[0]:.2f},{window[1]:.2f}]" if window else "nan"

        results.append((N, d_fix, d_adap))

        print(f"  {N:6d}  {2*N:6d}  {r_min_actual:10.5f}  {r_mean_actual:10.5f}"
              f"  {d_fix:12.4f}  {r2_fix:10.4f}"
              f"  {d_adap:13.4f}  {win_str:>12s}")

    print()
    print("=" * 92)
    print("  Summary (fixed-window)")
    print("=" * 92)
    print()
    dfix = np.array([r[1] for r in results])
    dfix = dfix[~np.isnan(dfix)]
    print(f"  mean   = {dfix.mean():.4f}")
    print(f"  std    = {dfix.std():.4f}")
    print(f"  min    = {dfix.min():.4f}")
    print(f"  max    = {dfix.max():.4f}")
    print(f"  deviation from 2.0 (mean) = {abs(dfix.mean() - 2.0):.4f}")
    print()
    if len(dfix) >= 3:
        tail = dfix[-3:]
        print(f"  last-3 std: {tail.std():.4f}")
        if tail.std() < 0.05:
            print("  → Converged to within 0.05")
        elif tail.std() < 0.10:
            print("  → Marginal convergence (std ~ 0.1)")
        else:
            print("  → NOT converged")
    print()
    print("  Target: d_spec → 2.0")
    print("=" * 92)


if __name__ == "__main__":
    main()