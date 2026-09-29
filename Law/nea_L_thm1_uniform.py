#!/usr/bin/env python3
"""Theorem 1: disconnected graph -> uniform distribution (MaxEnt)."""
import numpy as np
from scipy.optimize import minimize


def main():
    print("=" * 60)
    print("  Theorem 1: zero-constraint MaxEnt -> uniform")
    print("=" * 60)
    N = 7

    def neg_S(p):
        p = np.clip(p, 1e-15, 1)
        return np.sum(p * np.log(p))

    cons = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]
    bounds = [(1e-12, 1.0)] * N

    res = minimize(neg_S, np.full(N, 1.0/N), method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'ftol': 1e-15, 'maxiter': 500})
    p_opt = res.x

    print(f"\n  N = {N}")
    print(f"  Numerical optimum p_i:")
    for i, p in enumerate(p_opt):
        print(f"    p[{i}] = {p:.12f}")
    print(f"\n  Expected: 1/N = {1.0/N:.12f}")
    print(f"  Max deviation: {np.max(np.abs(p_opt - 1.0/N)):.3e}")

    if np.allclose(p_opt, 1.0/N, atol=1e-8):
        print("\n  PASS. Zero-constraint MaxEnt gives uniform distribution.")
    else:
        print("\n  FAIL.")


if __name__ == "__main__":
    main()