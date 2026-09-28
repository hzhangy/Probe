#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_micro_action_tangent_v2.py

Collective spectrum from the micro action:

    S = - sum_{i<j} Delta / (4 pi r_ij)

Projected onto TANGENT space (2N dim). Rotation zero modes removed
via QR projection. Adaptive Weyl fit (highest R^2 window).

Fixes:
  1. Robust tangent basis (choose reference axis with smallest |p_i|).
  2. Zero-mode projection via QR, not Gram-Schmidt on possibly
     nearly-degenerate rotation vectors.
  3. Sliding-window Weyl fit with R^2 score, not fixed window.
"""

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fibonacci_sphere(N):
    i = np.arange(N)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / max(1, N - 1)
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden * i
    return np.column_stack([r * np.cos(theta), y, r * np.sin(theta)])


def local_tangent_basis(p):
    """
    Orthonormal tangent vectors at p on the unit sphere.
    Reference axis = component with smallest |p_i|, so |e1| >= sqrt(2/3).
    """
    p = p / np.linalg.norm(p)
    idx = int(np.argmin(np.abs(p)))
    ref = np.zeros(3)
    ref[idx] = 1.0
    e1 = ref - np.dot(ref, p) * p
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(p, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def build_tangent_hessian(positions, Delta=0.133975):
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
            if r < 1e-12:
                continue
            r_hat = r_vec / r
            blk3 = (np.eye(3) - 3.0 * np.outer(r_hat, r_hat)) / r**3
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
    """
    Project out so(3) rotations.  QR on the (2N x 3) rotation matrix.
    """
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
    Q, _ = np.linalg.qr(R)  # 2N x 3
    P = np.eye(2 * N) - Q @ Q.T
    return P @ H @ P


def adaptive_weyl_slope(evals, min_points=30):
    """
    Sliding-window power-law fit N(λ) ~ λ^s.
    Returns (s_best, (frac_lo, frac_hi, R^2)).
    """
    ev = np.sort(evals)
    ev_max = np.max(np.abs(ev))
    ev = ev[ev > 1e-8 * ev_max]
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
    return best[1], best[2]


def main():
    print("=" * 84)
    print("  Micro-action Hessian, tangent space, adaptive Weyl fit")
    print("=" * 84)
    print()
    print("  Action:  S = - sum_{i<j} Delta / (4 pi r_ij)")
    print("  Dim:     2N (tangent space)  |  zero modes projected out")
    print()

    print(f"  {'N':>6s}  {'dim':>6s}  {'d_spec':>8s}  {'R^2':>8s}  {'window':>16s}")
    print("-" * 84)

    results = []
    for N in [48, 96, 192, 384, 768, 1536]:
        pos = fibonacci_sphere(N)
        H = build_tangent_hessian(pos)
        H = remove_rotation_zero_modes(H, pos)

        evals = la.eigvalsh(H)
        s, window = adaptive_weyl_slope(evals)
        d_spec = 2.0 * s if not np.isnan(s) else float("nan")
        if window is not None:
            r2 = window[2]
            win_str = f"[{window[0]:.2f}, {window[1]:.2f}]"
        else:
            r2 = float("nan")
            win_str = "nan"

        results.append((N, d_spec, r2))
        print(f"  {N:6d}  {2*N:6d}  {d_spec:8.4f}  {r2:8.4f}  {win_str:>16s}")

    print()
    print("=" * 84)
    print("  Target: d_spec -> 2.0")
    print("=" * 84)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    Ns = [r[0] for r in results]
    ds = [r[1] for r in results]

    ax = axes[0]
    ax.semilogx(Ns, ds, "o-", color="C0")
    ax.axhline(2.0, color="r", linestyle="--", label="2D target")
    ax.set_xlabel("N")
    ax.set_ylabel("d_spec")
    ax.set_title("Collective spectral dimension vs N")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Weyl law at largest N
    pos = fibonacci_sphere(results[-1][0])
    H = build_tangent_hessian(pos)
    H = remove_rotation_zero_modes(H, pos)
    evals = np.sort(la.eigvalsh(H))
    evals = evals[evals > 1e-8 * np.max(np.abs(evals))]
    counts = np.arange(1, len(evals) + 1)

    ax = axes[1]
    ax.loglog(evals, counts, "b-", alpha=0.75)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$N(\lambda)$")
    ax.set_title(f"Weyl law, N = {results[-1][0]}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("w_micro_action_tangent_v2.png", dpi=150)
    print("\n  Plot saved to w_micro_action_tangent_v2.png")


if __name__ == "__main__":
    main()