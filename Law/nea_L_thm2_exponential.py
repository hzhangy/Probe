#!/usr/bin/env python3
"""Theorem 2: 1D chain -> negative exponential (MaxEnt)."""
import numpy as np
from scipy.optimize import minimize


def main():
    print("=" * 60)
    print("  Theorem 2: 1D chain -> exponential (MaxEnt)")
    print("=" * 60)
    E = np.linspace(0.0, 5.0, 30)
    dE = E[1] - E[0]
    E_target = 1.0

    def neg_S(p):
        p_safe = np.clip(p, 1e-15, None)
        return np.sum(p_safe * np.log(p_safe))

    cons = [
        {'type': 'eq', 'fun': lambda p: np.sum(p * dE) - 1.0},
        {'type': 'eq', 'fun': lambda p: np.sum(p * E * dE) - E_target},
    ]
    bounds = [(1e-15, None)] * len(E)

    p0 = np.exp(-E)
    p0 /= np.sum(p0 * dE)
    res = minimize(neg_S, p0, method='SLSQP', bounds=bounds,
                   constraints=cons,
                   options={'ftol': 1e-14, 'maxiter': 2000})
    p_opt = res.x

    # Fit log p vs E
    valid = p_opt > 1e-12
    slope, intercept = np.polyfit(E[valid], np.log(p_opt[valid]), 1)
    beta_fit = -slope

    p_exp = beta_fit * np.exp(-beta_fit * E)
    p_exp /= np.sum(p_exp * dE)
    max_err = np.max(np.abs(p_opt - p_exp))

    print(f"\n  E grid points: {len(E)}")
    print(f"  <E> target: {E_target}")
    print(f"  Fitted beta: {beta_fit:.10f}")
    print(f"  Max deviation from exponential: {max_err:.3e}")

    if max_err < 1e-6:
        print("\n  PASS. 1D chain MaxEnt gives exponential distribution.")
    else:
        print("\n  FAIL.")


if __name__ == "__main__":
    main()