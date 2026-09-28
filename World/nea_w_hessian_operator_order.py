#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_w_hessian_operator_order.py

Determine the operator order m of the tangent-space Hessian
constructed from the micro action S = -sum Delta/(4 pi r_ij).

Diagnostic: |lambda_min| ~ N_faces^{-alpha}
  alpha = 0.5  -> m = 1 (first-order)
  alpha = 1.0  -> m = 2 (second-order)
  alpha = 1.5  -> m = 3 (third-order)
  ...

Also: Weyl slope s_D. If m=1, d_spec = s. If m=2, d_spec = 2s.
"""

import numpy as np
import scipy.linalg as la
from concurrent.futures import ProcessPoolExecutor, as_completed


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
        return np.array([1., 0., 0.]), np.array([0., 1., 0.])
    e1 /= n1
    e2 = np.cross(p, e1)
    return e1, e2


def build_tangent_hessian(positions, Delta=0.133975, r_min=0.01):
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
            if r < r_min:
                continue
            r_hat = r_vec / r
            blk3 = (np.eye(3) - 3.0 * np.outer(r_hat, r_hat)) / r ** 3
            blk3 *= D
            blk2 = T[i].T @ blk3 @ T[j]
            ii = slice(2*i, 2*i+2)
            jj = slice(2*j, 2*j+2)
            H[ii, jj] += -blk2
            H[jj, ii] += -blk2
            H[ii, ii] += blk2
            H[jj, jj] += blk2
    return 0.5 * (H + H.T)


def remove_rotation_zero_modes(H, positions):
    N = positions.shape[0]
    axes = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
    R = np.zeros((2 * N, 3))
    for a_idx, ax in enumerate(axes):
        for i in range(N):
            p = positions[i]
            e1, e2 = local_tangent_basis(p)
            dp = np.cross(ax, p)
            R[2*i, a_idx] = np.dot(dp, e1)
            R[2*i+1, a_idx] = np.dot(dp, e2)
    Q, _ = np.linalg.qr(R)
    P = np.eye(2 * N) - Q @ Q.T
    return 0.5 * ((P @ H @ P) + (P @ H @ P).T)


def weyl_slope(evals, fit_lo=0.10, fit_hi=0.60):
    ev = np.sort(np.abs(evals))
    ev = ev[ev > 1e-10 * np.max(ev)]
    n = len(ev)
    i0, i1 = int(fit_lo * n), int(fit_hi * n)
    s, _ = np.polyfit(np.log(ev[i0:i1]), np.log(np.arange(1, n+1)[i0:i1]), 1)
    return s


def worker(N):
    """Top-level for ProcessPoolExecutor."""
    pos = fibonacci_sphere(N)
    H = build_tangent_hessian(pos)
    H = remove_rotation_zero_modes(H, pos)

    ev = la.eigvalsh(H)
    ev_abs = np.sort(np.abs(ev))
    ev_nz = ev_abs[ev_abs > 1e-10 * np.max(ev_abs)]

    lambda_min = float(ev_nz[0])
    s_H = float(weyl_slope(ev))

    # H^2 slope
    mu = np.sort(ev_abs ** 2)
    mu = mu[mu > 1e-20 * np.max(mu)]
    n_mu = len(mu)
    i0, i1 = int(0.10 * n_mu), int(0.60 * n_mu)
    s_H2, _ = np.polyfit(
        np.log(mu[i0:i1]),
        np.log(np.arange(1, n_mu + 1)[i0:i1]),
        1,
    )

    return {
        'N': N,
        'dim': 2 * N,
        'lambda_min': lambda_min,
        's_H': s_H,
        's_H2': float(s_H2),
    }


def main():
    print("=" * 90)
    print("  Tangent-space Hessian: what is its operator order m?")
    print("=" * 90)
    print()

    Ns = [48, 96, 192, 384, 768, 1536]
    results = {}

    with ProcessPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(worker, N): N for N in Ns}
        for fut in as_completed(futures):
            N = futures[fut]
            try:
                r = fut.result()
                results[N] = r
                print("  N = {} done".format(N), flush=True)
            except Exception as e:
                print("  N = {} FAILED: {}".format(N, e), flush=True)

    print()
    data = [results[N] for N in sorted(results)]
    print("  {:>6s}  {:>6s}  {:>14s}  {:>10s}  {:>10s}  {:>10s}".format(
        "N", "dim", "lambda_min", "s_H", "s_H2", "s_H/s_H2"))
    print("  " + "-" * 70)
    for d in data:
        ratio = d['s_H'] / d['s_H2'] if d['s_H2'] > 0 else float('nan')
        print("  {:>6d}  {:>6d}  {:>14.6e}  {:>10.4f}  {:>10.4f}  {:>10.4f}".format(
            d['N'], d['dim'], d['lambda_min'],
            d['s_H'], d['s_H2'], ratio))
    print()

    # alpha from lambda_min scaling
    Ns_arr = np.array([d['N'] for d in data], dtype=float)
    lms = np.array([d['lambda_min'] for d in data])
    slope, _ = np.polyfit(np.log(Ns_arr), np.log(lms), 1)
    alpha = -slope

    print("  [lambda_min scaling]")
    print("    alpha = {:.4f}   (|lambda_min| ~ N^-alpha)".format(alpha))
    print()
    print("    Interpretation:")
    print("      alpha ~ 0.5  -> m = 1 (first-order)")
    print("      alpha ~ 1.0  -> m = 2 (second-order)")
    print("      alpha ~ 1.5  -> m = 3 (third-order)")
    print()

    # dimension analysis
    print("  [dimension analysis of H_ij]")
    print("    H_ij = -Delta/(4 pi) * (delta_ab/r^3 - 3 r_hat_a r_hat_b / r^3)")
    print("    [H_ij] = [1/L^3]")
    print()
    print("    BUT: H acts on DISPLACEMENTS delta x_i (dimension [L]).")
    print("    Output is FORCE (dimension [E/L] = [ML/T^2]).")
    print("    Physical interpretation: H is a STIFFNESS MATRIX.")
    print("    Continuous analog: second functional derivative of 1/r energy.")
    print("    Known result: on sphere, 1/r kernel's tangent Hessian is a")
    print("    first-order pseudodifferential operator (order 1).")
    print()

    # verdict
    print("=" * 90)
    print("  VERDICT")
    print("=" * 90)
    print()
    if abs(alpha - 0.5) < abs(alpha - 1.0) and abs(alpha - 0.5) < abs(alpha - 1.5):
        m_est = 1
    elif abs(alpha - 1.0) < abs(alpha - 1.5):
        m_est = 2
    else:
        m_est = 3
    print("  From alpha = {:.4f}: m_est = {}".format(alpha, m_est))
    print()
    print("  If m = 1: d_spec = s_H (NOT 2 s_H)")
    print("  If m = 2: d_spec = 2 s_H")
    print()
    print("  Final d_spec values (under both hypotheses):")
    print()
    print("  {:>6s}  {:>10s}  {:>12s}  {:>12s}".format(
        "N", "s_H", "if m=1", "if m=2"))
    print("  " + "-" * 46)
    for d in data:
        print("  {:>6d}  {:>10.4f}  {:>12.4f}  {:>12.4f}".format(
            d['N'], d['s_H'], d['s_H'], 2 * d['s_H']))


if __name__ == "__main__":
    main()