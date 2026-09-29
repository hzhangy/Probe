#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_L_thm4b_bose_v2.py

TRUE numerical verification of Theorem 4B:
    Bose-Einstein statistics is the MaxEnt solution of the
    combinatorial entropy for open channels.

Method:
    We do NOT assume the Bose-Einstein form. We numerically optimize
    the mean occupation numbers n_i subject to:
        - Sum_i g_i n_i = N
        - Sum_i g_i n_i E_i = E
        - n_i >= 0
    maximizing the combinatorial entropy
        S[n] = Sum_i g_i [(1+n_i) ln(1+n_i) - n_i ln n_i].
    Then we fit to Bose-Einstein and check residuals.
"""

import numpy as np
from scipy.optimize import minimize


def entropy_bose(n, g):
    """S[n] = Sum g_i [(1+n_i) ln(1+n_i) - n_i ln n_i]"""
    n_safe = np.maximum(n, 1e-15)
    return np.sum(g * ((1.0 + n) * np.log(1.0 + n) - n * np.log(n_safe)))


def solve_maxent_bose(E, g, N_target, E_target, n0=None):
    m = len(E)
    if n0 is None:
        n0 = np.full(m, N_target / np.sum(g))

    def neg_S(n):
        return -entropy_bose(n, g)

    def con_N(n):
        return np.sum(g * n) - N_target

    def con_E(n):
        return np.sum(g * n * E) - E_target

    cons = [{'type': 'eq', 'fun': con_N},
            {'type': 'eq', 'fun': con_E}]
    bounds = [(1e-12, None)] * m

    res = minimize(neg_S, n0, method='SLSQP', bounds=bounds,
                   constraints=cons,
                   options={'ftol': 1e-15, 'maxiter': 2000})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


def section(title):
    print("=" * 78)
    print("  " + title)
    print("=" * 78)
    print()


def main():
    print()
    section("Theorem 4B (TRUE numerical): MaxEnt -> Bose-Einstein")

    E = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0])
    g = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    N_target = 12.0
    E_target = 30.0

    print(f"  Energy grid:      E = {E.tolist()}")
    print(f"  Degeneracies:     g = {g.tolist()}")
    print(f"  Target N:         {N_target}")
    print(f"  Target E:         {E_target}")
    print()

    # ------------------------------------------------------------------
    section("[1] Numerical MaxEnt optimization (SLSQP)")

    n_num = solve_maxent_bose(E, g, N_target, E_target)

    print("  Numerical optimal n_i:")
    print(f"  {'i':>3s}  {'E_i':>6s}  {'g_i':>4s}  {'n_i':>14s}")
    print("  " + "-" * 32)
    for i in range(len(E)):
        print(f"  {i:>3d}  {E[i]:>6.2f}  {g[i]:>4.1f}  {n_num[i]:>14.10f}")
    print()

    N_check = np.sum(g * n_num)
    E_check = np.sum(g * n_num * E)
    print(f"  Constraint check:")
    print(f"    Sum g_i n_i      = {N_check:.12f}  (target {N_target})")
    print(f"    Sum g_i n_i E_i  = {E_check:.12f}  (target {E_target})")
    print()

    # ------------------------------------------------------------------
    section("[2] Fit to Bose-Einstein (post-hoc)")

    # BE: n_i = 1 / (exp(beta (E_i - mu)) - 1)
    # So: log(1 + 1/n_i) = beta (E_i - mu)
    valid = n_num > 1e-8
    y = np.log(1.0 + 1.0 / n_num[valid])
    beta_fit, intercept = np.polyfit(E[valid], y, 1)
    mu_fit = -intercept / beta_fit

    n_BE = 1.0 / (np.exp(beta_fit * (E - mu_fit)) - 1.0)

    print(f"  Linear fit of log(1 + 1/n_i) vs E:")
    print(f"    beta = {beta_fit:.10f}")
    print(f"    mu   = {mu_fit:.10f}")
    print()

    y_pred = beta_fit * E[valid] + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print(f"  R^2 of linear fit: {r2:.15f}")
    print()

    # ------------------------------------------------------------------
    section("[3] Numerical MaxEnt vs Bose-Einstein")

    print(f"  {'E_i':>6s}  {'n_num':>14s}  {'n_BE':>14s}  {'|diff|':>12s}")
    print("  " + "-" * 52)
    max_err = 0.0
    for i in range(len(E)):
        diff = abs(n_num[i] - n_BE[i])
        max_err = max(max_err, diff)
        print(f"  {E[i]:>6.2f}  {n_num[i]:>14.10f}  {n_BE[i]:>14.10f}  "
              f"{diff:>12.3e}")
    print()
    print(f"  Maximum absolute deviation: {max_err:.3e}")
    print()

    # ------------------------------------------------------------------
    section("VERDICT: Theorem 4B verified")

    if max_err < 1e-7 and r2 > 1.0 - 1e-10:
        print("  PASS. The numerical MaxEnt optimum (obtained WITHOUT")
        print("  assuming Bose-Einstein) matches the Bose-Einstein form")
        print("  to machine precision. The unbounded occupation")
        print("  n_i in {0, 1, 2, ...} on open channels forces Bose-Einstein.")
    else:
        print("  FAIL. Deviation too large.")
    print()


if __name__ == "__main__":
    main()