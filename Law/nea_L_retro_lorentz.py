#!/usr/bin/env python3
"""TCVP re-derivation of Lorentz violation: delta_omega^2 = -k^4 a^2/12."""
import numpy as np


def main():
    print("=" * 60)
    print("  TCVP re-derivation of Lorentz violation")
    print("=" * 60)

    a = 1.0
    kappa = 1.0 / a**2  # Lorentz-recovery condition

    print(f"\n  Lattice spacing a = {a}")
    print(f"  kappa * a^2 = 1 (Lorentz recovery)")
    print()

    k_values = np.array([0.02, 0.05, 0.1, 0.2, 0.5])
    print(f"  {'k':>8s}  {'exact':>16s}  {'-k^4 a^2/12':>16s}  "
          f"{'rel. dev':>12s}")
    print("  " + "-" * 60)

    for k in k_values:
        # Exact: 2 kappa [1 - cos(k a)]
        omega2_exact = 2 * kappa * (1 - np.cos(k * a))
        omega2_cont = k**2
        delta_exact = omega2_exact - omega2_cont
        delta_theory = -k**4 * a**2 / 12
        rel_dev = abs(delta_exact - delta_theory) / abs(delta_theory) \
                  if delta_theory != 0 else 0
        print(f"  {k:>8.4f}  {delta_exact:>16.10e}  "
              f"{delta_theory:>16.10e}  {rel_dev:>12.4f}")

    # Scaling exponent in k
    k_arr = np.logspace(-2, -0.3, 30)
    delta_exact = 2 * kappa * (1 - np.cos(k_arr * a)) - k_arr**2
    delta_theory = -k_arr**4 * a**2 / 12
    valid = np.abs(delta_theory) > 1e-15
    slope, _ = np.polyfit(np.log(k_arr[valid]),
                          np.log(np.abs(delta_exact[valid])), 1)

    print(f"\n  Measured k scaling exponent: {slope:.6f}")
    print(f"  Theoretical: 4")
    print(f"  Deviation: {abs(slope - 4):.4f}")

    if abs(slope - 4) < 0.01:
        print("\n  PASS. delta_omega^2 = -k^4 a^2/12 confirmed.")
    else:
        print("\n  FAIL.")


if __name__ == "__main__":
    main()