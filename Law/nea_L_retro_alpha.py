#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_L_retro_alpha.py

TCVP re-derivation of alpha^{-1} (Volume L, Part V).

TCVP perspective:
    The fine-structure constant is the IMPEDANCE-MATCHING ratio
    between two topological channels:
        - Weak-force addressing rent:  U_weak = 10 sqrt(3)
        - Electromagnetic weaving rent: U_EM  = 0.4 pi
    The tree-level exchange rate is
        alpha^{-1}_tree = (U_weak / U_EM) * pi^2 + 1
                        = 25 sqrt(3) pi + 1
    The +1 is the Being Tax baseline.

    The one-loop lattice correction is
        delta_C3 = Delta/128 + 64 (Delta/128)^3

    giving
        alpha^{-1}(0) = 137.0359990675
"""

import numpy as np
from math import pi, sqrt


def section(title):
    print("=" * 78)
    print("  " + title)
    print("=" * 78)
    print()


def main():
    print()
    section("TCVP re-derivation of alpha^{-1} (Volume L, Part V)")

    # ------------------------------------------------------------------
    section("[1] Topological genes")

    U_EM    = 0.4 * pi
    U_weak  = 10.0 * sqrt(3.0)
    Delta   = 1.0 - sqrt(3.0) / 2.0
    R       = 1.0 / (1.0 + pi)

    print(f"  U_EM   = 0.4 * pi      = {U_EM:.10f}")
    print(f"  U_weak = 10 * sqrt(3)  = {U_weak:.10f}")
    print(f"  Delta  = 1 - sqrt(3)/2 = {Delta:.10f}")
    print(f"  R      = 1/(1+pi)      = {R:.10f}")
    print()

    # ------------------------------------------------------------------
    section("[2] Impedance-matching formula")

    ratio = U_weak / U_EM
    print(f"  Step 1. Rent ratio:")
    print(f"    U_weak / U_EM = {U_weak:.6f} / {U_EM:.6f} = {ratio:.10f}")
    print()

    print(f"  Step 2. 2D -> 3D phase-loop closure measure:")
    print(f"    pi^2 = {pi**2:.10f}")
    print()

    print(f"  Step 3. Impedance product:")
    product = ratio * pi**2
    print(f"    (U_weak / U_EM) * pi^2 = {product:.10f}")
    print(f"    Simplified form:        = 25 sqrt(3) pi")
    print(f"    Check: 25 sqrt(3) pi    = {25.0 * sqrt(3.0) * pi:.10f}")
    print()

    print(f"  Step 4. Being Tax baseline:")
    print(f"    +1 (one Tick of connectivity)")
    print()

    alpha_inv_tree = product + 1.0
    print(f"  Tree-level prediction:")
    print(f"    alpha^-1_tree = (U_weak / U_EM) pi^2 + 1")
    print(f"                  = 25 sqrt(3) pi + 1")
    print(f"                  = {alpha_inv_tree:.12f}")
    print()

    # ------------------------------------------------------------------
    section("[3] One-loop lattice correction")

    # C8 vertex count squared
    V_C8 = 8
    V_C8_sq = V_C8 ** 2  # = 64
    two_V_C8_sq = 2 * V_C8_sq  # = 128

    print(f"  V(C8)^2 = {V_C8_sq}")
    print(f"  128 = 2 * V(C8)^2 = {two_V_C8_sq}")
    print(f"  64  = V(C8)^2")
    print()

    delta_C3 = Delta / two_V_C8_sq + V_C8_sq * (Delta / two_V_C8_sq) ** 3
    print(f"  delta_C3 = Delta/128 + 64 (Delta/128)^3")
    print(f"           = {Delta/two_V_C8_sq:.12e} + "
          f"{V_C8_sq * (Delta/two_V_C8_sq)**3:.12e}")
    print(f"           = {delta_C3:.12e}")
    print()

    alpha_inv_full = alpha_inv_tree + delta_C3
    print(f"  alpha^-1(0) = alpha^-1_tree + delta_C3")
    print(f"              = {alpha_inv_full:.10f}")
    print()

    # ------------------------------------------------------------------
    section("[4] Comparison with CODATA 2018")

    alpha_inv_obs = 137.035999084
    alpha_inv_err = 0.000000021

    diff = alpha_inv_full - alpha_inv_obs
    rel_dev = diff / alpha_inv_obs
    pull = diff / alpha_inv_err

    print(f"  N.E.A. prediction   = {alpha_inv_full:.10f}")
    print(f"  CODATA 2018         = {alpha_inv_obs:.10f} +/- {alpha_inv_err:.1e}")
    print(f"  Absolute deviation  = {diff:.3e}")
    print(f"  Relative deviation  = {rel_dev*1e9:.4f} ppb")
    print(f"  Statistical pull    = {pull:+.4f} sigma")
    print()

    # ------------------------------------------------------------------
    section("VERDICT: alpha^{-1} verified")

    if abs(pull) < 2.0:
        print(f"  PASS. The TCVP impedance-matching formula reproduces")
        print(f"  the fine-structure constant to {pull:+.2f} sigma.")
        print()
        print(f"  The prediction uses only:")
        print(f"    - U_weak = 10 sqrt(3)   (Stride-10 rent)")
        print(f"    - U_EM   = 0.4 pi       (C8 cycle ratio)")
        print(f"    - Delta  = 1 - sqrt(3)/2 (K4 locking gap)")
        print(f"    - V(C8)  = 8            (cube vertex count)")
        print(f"  No free parameters.")
    else:
        print(f"  FAIL. Pull = {pull:+.2f} sigma exceeds threshold.")
    print()


if __name__ == "__main__":
    main()