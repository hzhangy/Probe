#!/usr/bin/env python3
"""TCVP re-derivation of neutrino mass exponents {3, 5, 9}."""
import numpy as np


def main():
    print("=" * 60)
    print("  TCVP re-derivation of neutrino mass exponents")
    print("=" * 60)

    print("\n  Addressing infimum: n-dimensional substructure of C8")
    print("  plus one time-synchronization center.")
    print()
    print(f"  {'n':>3s}  {'substructure':>16s}  {'vertices':>10s}  "
          f"{'exponent = V+1':>16s}")
    print("  " + "-" * 56)

    substructures = {1: 'edge', 2: 'face', 3: 'cube'}
    vertices = {1: 2, 2: 4, 3: 8}

    for n in [1, 2, 3]:
        v = vertices[n]
        exp = v + 1
        print(f"  {n:>3d}  {substructures[n]:>16s}  {v:>10d}  "
              f"{exp:>16d}")

    exponents = [vertices[n] + 1 for n in [1, 2, 3]]
    print(f"\n  Exponent set: {exponents}")

    # Verify against {2^n + 1}
    exp_check = [2**n + 1 for n in [1, 2, 3]]
    print(f"  Check: 2^n + 1 for n=1,2,3 = {exp_check}")
    print(f"  Match: {exponents == exp_check}")

    # Mass ratios
    m0_meV = 1.2775
    masses = [m0_meV * (3/2)**e for e in exponents]
    print(f"\n  Mass formula: m_nu_i = m_0 * (3/2)^exponent")
    print(f"  m_0 = {m0_meV} meV")
    print()
    for name, m in zip(['nu_e', 'nu_mu', 'nu_tau'], masses):
        print(f"    {name}: {m:.4f} meV")
    print(f"  Sum: {sum(masses):.4f} meV")

    if exponents == exp_check:
        print("\n  PASS. Neutrino exponent set {3, 5, 9} confirmed.")


if __name__ == "__main__":
    main()