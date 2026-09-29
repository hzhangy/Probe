#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_L_retro_d3_throughput.py

TCVP re-derivation of d = 3 (Volume L, Part V).

TCVP perspective:
    d = 3 is the UNIQUE maximizer of the information throughput
        J(d) = C_eff(d) * F(d)
    where:
        C_eff(d)    = max(0, 1 - C_coord(d))
        C_coord(d)  = (1/(1+pi)) * binom(d, 2)
        F(d)        = number of transverse propagating gravitational
                      degrees of freedom in (d+1)-dimensional spacetime
                    = (d+1)*(d-2)/2   (massless graviton, D = d+1)

Verification:
    - d=1: J=0 (no transverse)
    - d=2: J=0 (Mermin-Wagner; 2+1D GR has no propagating Weyl)
    - d=3: J > 0 (UNIQUE positive throughput)
    - d=4: C_eff(4) < 0, J=0
    - d>=5: J=0
"""

import numpy as np
from math import comb, pi


def C_coord(d):
    """Coordination overhead = eta * binom(d, 2), eta = 1/(1+pi)."""
    return (1.0 / (1.0 + pi)) * comb(d, 2)


def C_eff(d):
    """Effective channel capacity = max(0, 1 - C_coord)."""
    return max(0.0, 1.0 - C_coord(d))


def F_transverse(d):
    """
    Number of transverse propagating graviton polarizations in
    (d+1)-dimensional spacetime:
        D = d + 1
        F = D(D-3)/2
    d=2: 0   (2+1D GR, no Weyl tensor)
    d=3: 2   (3+1D, two tensor polarizations)
    d=4: 5   (4+1D)
    d=5: 9   (5+1D)
    """
    if d < 2:
        return 0
    D = d + 1
    return D * (D - 3) // 2


def J_throughput(d):
    """Total information throughput."""
    return C_eff(d) * F_transverse(d)


def section(title):
    print("=" * 78)
    print("  " + title)
    print("=" * 78)
    print()


def main():
    print()
    section("TCVP re-derivation of d = 3 (Volume L, Part V)")

    # ------------------------------------------------------------------
    section("[1] Coordination tax across dimensions")
    print(f"  eta = 1/(1+pi) = {1.0/(1.0+pi):.10f}")
    print()
    print(f"  {'d':>3s}  {'C_coord(d)':>14s}  {'C_eff(d)':>14s}  "
          f"{'F(d)':>8s}  {'J(d)':>14s}")
    print("  " + "-" * 60)

    for d in range(1, 11):
        c_coord = C_coord(d)
        c_eff = C_eff(d)
        f_d = F_transverse(d)
        j_d = J_throughput(d)
        print(f"  {d:>3d}  {c_coord:>14.10f}  {c_eff:>14.10f}  "
              f"{f_d:>8d}  {j_d:>14.10f}")
    print()

    # ------------------------------------------------------------------
    section("[2] Uniqueness of d = 3")

    positive_dims = [d for d in range(1, 11) if J_throughput(d) > 0]
    print(f"  Dimensions with J(d) > 0: {positive_dims}")
    print(f"  Count: {len(positive_dims)}")
    print()

    if positive_dims == [3]:
        print("  PASS. d = 3 is the UNIQUE positive-throughput dimension.")
    else:
        print("  FAIL. Multiple dimensions have positive throughput.")
    print()

    # ------------------------------------------------------------------
    section("[3] Why d=1, d=2, d=4, d>=5 are excluded")

    print(f"  d = 1:  C_coord = {C_coord(1):.6f},  F = {F_transverse(1)},  "
          f"J = {J_throughput(1):.6f}")
    print(f"          Excluded: no transverse degrees of freedom (sterile).")
    print()
    print(f"  d = 2:  C_coord = {C_coord(2):.6f},  F = {F_transverse(2)},  "
          f"J = {J_throughput(2):.6f}")
    print(f"          Excluded: Mermin-Wagner forbids continuous symmetry")
    print(f"          breaking; 2+1D GR has zero propagating Weyl DoF.")
    print()
    print(f"  d = 3:  C_coord = {C_coord(3):.6f},  F = {F_transverse(3)},  "
          f"J = {J_throughput(3):.10f}")
    print(f"          SOLVENT: C_eff = {C_eff(3):.6f} > 0, F = 2.")
    print()
    print(f"  d = 4:  C_coord = {C_coord(4):.6f},  F = {F_transverse(4)},  "
          f"J = {J_throughput(4):.6f}")
    print(f"          Excluded: C_coord = {C_coord(4):.6f} > 1, "
          f"instantaneous phase decoherence.")
    print()
    print(f"  d >= 5: C_coord monotonically increases, "
          f"C_eff = 0 for all d >= 4.")
    print()

    # ------------------------------------------------------------------
    section("[4] Marginal surplus at d = 3")

    surplus = 1.0 - C_coord(3)
    print(f"  C_coord(3)          = {C_coord(3):.10f}")
    print(f"  Marginal surplus    = 1 - C_coord(3) = {surplus:.10f}")
    print(f"  Physical meaning:   the residual bandwidth available")
    print(f"                      for internal state transitions and")
    print(f"                      gravitational wave propagation.")
    print()

    # ------------------------------------------------------------------
    section("VERDICT: d = 3 uniquely selected")

    if positive_dims == [3] and surplus > 0:
        print("  PASS. Under the Being Tax B = 1 and the coordination")
        print("  overhead C_coord(d) = (1/(1+pi)) binom(d, 2), the")
        print("  information throughput J(d) is positive ONLY at d = 3.")
        print()
        print("  The TCVP derivation reproduces the Volume B solvency")
        print("  theorem independently, using only the throughput")
        print("  maximization principle.")
    else:
        print("  FAIL.")
    print()


if __name__ == "__main__":
    main()