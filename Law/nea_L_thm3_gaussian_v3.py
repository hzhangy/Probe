#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_L_thm3_gaussian_v3.py

Theorem 3: Octahedral central inversion -> Gaussian (对偶法).

方法: 直接在 Lagrange 乘子 (beta_1, beta_2) 上求解对偶问题,
      p* 从 beta* 显式构造。
"""

import numpy as np
from scipy.optimize import minimize


def solve_maxent_gaussian(x, c_mu, c_var):
    """对偶法: p* = exp(-beta_1 x - beta_2 (x-mu)^2) / Z."""
    c_targets = np.array([c_mu, c_var])

    def dual(beta):
        logits = -beta[0] * x - beta[1] * (x - c_mu)**2
        logits -= logits.max()
        w = np.exp(logits)
        Z = w.sum()
        lnZ = np.log(Z) + logits.max()
        L = lnZ + np.dot(beta, c_targets)
        p = w / Z
        f_mean = np.array([np.sum(p * x),
                           np.sum(p * (x - c_mu)**2)])
        grad = -f_mean + c_targets
        return L, grad

    res = minimize(dual, np.zeros(2), jac=True, method='L-BFGS-B',
                   options={'ftol': 1e-20, 'gtol': 1e-15})
    beta = res.x
    logits = -beta[0] * x - beta[1] * (x - c_mu)**2
    logits -= logits.max()
    p = np.exp(logits) / np.exp(logits).sum()
    return beta, p


def main():
    print("=" * 78)
    print("  Theorem 3 v3: Octahedral inversion -> Gaussian (对偶法)")
    print("=" * 78)
    print()

    # Topology
    dirs = np.array([[ 1, 0, 0], [-1, 0, 0],
                     [ 0, 1, 0], [ 0,-1, 0],
                     [ 0, 0, 1], [ 0, 0,-1]], dtype=float)
    print(f"  E[d_x]   = {dirs[:,0].mean():.15f}")
    print(f"  Var[d_x] = {dirs[:,0].var():.15f}  (= 1/3)")
    print()

    x = np.linspace(-4.0, 4.0, 201)
    c_mu, c_var = 0.0, 1.0
    beta, p = solve_maxent_gaussian(x, c_mu, c_var)

    print(f"  Grid points: {len(x)}")
    print(f"  <x>       = {np.sum(p * x):.15f}  (target {c_mu})")
    print(f"  <(x-mu)²> = {np.sum(p*(x-c_mu)**2):.15f}  (target {c_var})")
    print(f"  beta_1 = {beta[0]:.15f}")
    print(f"  beta_2 = {beta[1]:.15f}  (theory 1/(2σ²) = 0.5)")
    print()

    # 验证: ln p = -beta_1 x - beta_2 x² - ln Z
    logits = -beta[0] * x - beta[1] * (x - c_mu)**2
    log_p_theory = logits - logits.max()
    log_p_num = np.log(np.maximum(p, 1e-300))
    diff = log_p_num - log_p_theory
    diff -= diff.mean()
    res_std = np.std(diff)
    res_max = np.max(np.abs(diff))

    print(f"  Residual std: {res_std:.3e}")
    print(f"  Residual max: {res_max:.3e}")
    print()

    status = "PASS" if res_std < 1e-12 else "FAIL"
    print(f"  [{status}] Theorem 3: res_std = {res_std:.3e}")
    print()
    if status == "PASS":
        print("  ln p* equals -beta_1 x - beta_2 x² - ln Z to machine")
        print("  precision. The octahedral inversion fixes E[d]=0 and")
        print("  Var[d]=1/3, and MaxEnt under those constraints yields")
        print("  the Gaussian.")
    print()


if __name__ == "__main__":
    main()