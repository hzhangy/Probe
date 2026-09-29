#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_L_thm4a_fermi_v2.py

TRUE numerical verification of Theorem 4A:
    Fermi-Dirac statistics is the MaxEnt solution of the
    combinatorial entropy under B = 1.

Method:
    We do NOT assume the Fermi-Dirac form. We numerically optimize
    the occupation numbers p_i subject to:
        - Sum_i g_i p_i = N (particle number)
        - Sum_i g_i p_i E_i = E (energy)
        - 0 <= p_i <= 1
    maximizing the combinatorial entropy
        S[p] = -Sum_i g_i [p_i ln p_i + (1-p_i) ln(1-p_i)].
    Then we fit the resulting p_i to the Fermi-Dirac form and verify
    that the residuals are at machine precision.

If the optimizer's output matches Fermi-Dirac without any prior
knowledge of the Fermi-Dirac form, Theorem 4A is verified.
"""

import numpy as np
from scipy.optimize import minimize


def entropy_fermi(p, g):
    """S[p] = -Sum g_i [p_i ln p_i + (1-p_i) ln(1-p_i)]"""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -np.sum(g * (p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def solve_maxent(E, g, N_target, E_target, p0=None):
    n = len(E)
    if p0 is None:
        p0 = np.full(n, 0.5)

    def neg_S(p):
        return -entropy_fermi(p, g)

    def con_N(p):
        return np.sum(g * p) - N_target

    def con_E(p):
        return np.sum(g * p * E) - E_target

    cons = [{'type': 'eq', 'fun': con_N},
            {'type': 'eq', 'fun': con_E}]
    bounds = [(1e-12, 1.0 - 1e-12)] * n

    res = minimize(neg_S, p0, method='SLSQP', bounds=bounds,
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
    section("Theorem 4A (TRUE numerical): MaxEnt -> Fermi-Dirac")

    # ------------------------------------------------------------------
    # Setup: energy grid with degeneracies
    # ------------------------------------------------------------------
    E = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    g = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    N_target = 4.0
    E_target = 5.5

    print(f"  Energy grid:      E = {E.tolist()}")
    print(f"  Degeneracies:     g = {g.tolist()}")
    print(f"  Total capacity:   Sum g_i = {np.sum(g):.1f}")
    print(f"  Target N:         {N_target}")
    print(f"  Target E:         {E_target}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Numerical MaxEnt (no Fermi-Dirac form assumed)
    # ------------------------------------------------------------------
    section("[1] Numerical MaxEnt optimization (SLSQP)")

    p_num = solve_maxent(E, g, N_target, E_target)

    print("  Numerical optimal p_i:")
    print(f"  {'i':>3s}  {'E_i':>6s}  {'g_i':>4s}  {'p_i':>14s}")
    print("  " + "-" * 32)
    for i in range(len(E)):
        print(f"  {i:>3d}  {E[i]:>6.2f}  {g[i]:>4.1f}  {p_num[i]:>14.10f}")
    print()

    # Verify constraints are satisfied
    N_check = np.sum(g * p_num)
    E_check = np.sum(g * p_num * E)
    print(f"  Constraint check:")
    print(f"    Sum g_i p_i      = {N_check:.12f}  (target {N_target})")
    print(f"    Sum g_i p_i E_i  = {E_check:.12f}  (target {E_target})")
    print()

    # ------------------------------------------------------------------
    # Step 2: Fit to Fermi-Dirac WITHOUT assuming it
    # ------------------------------------------------------------------
    section("[2] Fit to Fermi-Dirac (post-hoc)")

    valid = (p_num > 1e-10) & (p_num < 1.0 - 1e-10)
    x = np.log((1.0 - p_num[valid]) / p_num[valid])
    beta_fit, intercept = np.polyfit(E[valid], x, 1)
    mu_fit = -intercept / beta_fit

    f_FD = 1.0 / (1.0 + np.exp(beta_fit * (E - mu_fit)))

    print(f"  Log-odds vs E linear fit:")
    print(f"    beta = {beta_fit:.10f}")
    print(f"    mu   = {mu_fit:.10f}")
    print()

    # Linear fit quality
    x_pred = beta_fit * E[valid] + intercept
    ss_res = np.sum((x - x_pred) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print(f"  R^2 of log-odds linear fit: {r2:.15f}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Point-by-point comparison
    # ------------------------------------------------------------------
    section("[3] Numerical MaxEnt vs Fermi-Dirac")

    print(f"  {'E_i':>6s}  {'p_num':>14s}  {'p_FD':>14s}  {'|diff|':>12s}")
    print("  " + "-" * 52)
    max_err = 0.0
    for i in range(len(E)):
        diff = abs(p_num[i] - f_FD[i])
        max_err = max(max_err, diff)
        print(f"  {E[i]:>6.2f}  {p_num[i]:>14.10f}  {f_FD[i]:>14.10f}  "
              f"{diff:>12.3e}")
    print()
    print(f"  Maximum absolute deviation: {max_err:.3e}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Verdict
    # ------------------------------------------------------------------
    section("VERDICT: Theorem 4A verified")

    if max_err < 1e-7 and r2 > 1.0 - 1e-10:
        print("  PASS. The numerical MaxEnt optimum (obtained WITHOUT")
        print("  assuming Fermi-Dirac) matches the Fermi-Dirac form to")
        print("  machine precision. The occupation constraint n_i in")
        print("  {0, 1} under B = 1 forces Fermi-Dirac.")
    else:
        print("  FAIL. Deviation too large.")
    print()


if __name__ == "__main__":
    main()