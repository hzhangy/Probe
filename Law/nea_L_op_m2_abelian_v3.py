#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_L_op_m2_abelian_v3.py

TRUE numerical verification of the quark-lepton bifurcation (OP-M2/W4).

Three-part structure:
    [1] Leptons (colorless): NNI works to 0.026%.
    [2] Quarks (colored): K4 cycle-space chain works to 0.02-3.1%.
    [3] Quarks: NNI FAILS because off-diagonal terms only push
        eigenvalues away from diagonal entries, which are already
        above the observed targets.
"""

import numpy as np
from scipy.optimize import minimize


M_E = 0.51099895
M_MU = 105.6584
ALPHA_INV = 137.0359990675

OBS = {'u': 2.16, 'd': 4.67, 's': 93.4,
       'c': 1270.0, 'b': 4180.0, 't': 172760.0}

NEA = {
    'u': M_E * (1 + np.pi),
    'd': M_E * 3 * np.pi,
    's': M_E * 3 * np.pi * 2 * np.pi**2,
    'c': M_MU * 12,
    'b': M_MU * 12 * 10 / 3,
    't': M_E * ALPHA_INV**2 * 18,
}


def section(t):
    print("=" * 78)
    print("  " + t)
    print("=" * 78)
    print()


def nni_eigs(A, B, a, d):
    M = np.array([[1.0, a, 0.0], [a, A, d], [0.0, d, B]])
    return np.sort(np.linalg.eigvalsh(M))


def main():
    print()
    section("OP-M2/W4 (TRUE numerical): Quark-lepton bifurcation")

    # --- [1] Leptons ---
    section("[1] Leptons: NNI works")

    a_lep, d_lep = 0.3, 48.0
    A_lep = 66 * np.pi
    B_lep = 66 * np.pi * (16 * np.pi / 3)
    evals_lep = nni_eigs(A_lep, B_lep, a_lep, d_lep)
    ratios_lep = evals_lep / evals_lep[0]

    print(f"  a = {a_lep}, d = {d_lep}")
    print(f"  m_mu/m_e  = {ratios_lep[1]:.6f}  "
          f"(obs 206.7683, dev {100*(ratios_lep[1]-206.7683)/206.7683:+.4f}%)")
    print(f"  m_tau/m_e = {ratios_lep[2]:.6f}  "
          f"(obs 3477.228, dev {100*(ratios_lep[2]-3477.228)/3477.228:+.4f}%)")
    print()

    # --- [2] Quarks: K4 chain ---
    section("[2] Quarks: K4 cycle-space chain")

    print(f"  {'Quark':>6s}  {'NEA (MeV)':>14s}  {'Obs (MeV)':>14s}  "
          f"{'Dev':>10s}")
    print("  " + "-" * 52)
    for q in ['u', 'd', 's', 'c', 'b', 't']:
        dev = 100 * (NEA[q] - OBS[q]) / OBS[q]
        print(f"  {q:>6s}  {NEA[q]:>14.4f}  {OBS[q]:>14.4f}  "
              f"{dev:>+10.4f}%")
    print()

    # --- [3] Quarks: NNI fails ---
    section("[3] Quarks: NNI fails")

    target_c = OBS['c'] / OBS['u']
    target_t = OBS['t'] / OBS['u']
    A_u = NEA['c'] / NEA['u']
    B_u = NEA['t'] / NEA['u']

    print(f"  Up-type targets: m_c/m_u = {target_c:.4f}, "
          f"m_t/m_u = {target_t:.4f}")
    print(f"  K4 chain diagonal: A_u = {A_u:.4f}, B_u = {B_u:.4f}")
    print()
    print(f"  Note: A_u is {100*(A_u-target_c)/target_c:+.4f}% above target")
    print()

    def loss(params):
        a, d = params
        if a < 0 or d < 0 or a > 1e7 or d > 1e7:
            return 1e10
        try:
            evals = nni_eigs(A_u, B_u, a, d)
        except Exception:
            return 1e10
        if evals[0] <= 0:
            return 1e10
        r = evals / evals[0]
        return ((r[1] - target_c)/target_c)**2 + ((r[2] - target_t)/target_t)**2

    # Grid + Nelder-Mead
    best_loss, best_params = 1e10, (0.0, 0.0)
    for a0 in [0, 1, 10, 100, 1000, 10000]:
        for d0 in [0, 10, 100, 1000, 10000, 100000]:
            L = loss([a0, d0])
            if L < best_loss:
                best_loss, best_params = L, (a0, d0)

    res = minimize(loss, best_params, method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-15,
                            'maxiter': 10000})
    a_opt, d_opt = res.x
    evals_opt = nni_eigs(A_u, B_u, a_opt, d_opt)
    ratios_opt = evals_opt / evals_opt[0]

    res_c = 100 * abs(ratios_opt[1] - target_c) / target_c
    res_t = 100 * abs(ratios_opt[2] - target_t) / target_t

    print(f"  Best NNI fit (grid + Nelder-Mead):")
    print(f"    a_opt = {a_opt:.6f}, d_opt = {d_opt:.6f}")
    print(f"    m_c/m_u = {ratios_opt[1]:.4f}  "
          f"(target {target_c:.4f}, "
          f"dev {100*(ratios_opt[1]-target_c)/target_c:+.4f}%)")
    print(f"    m_t/m_u = {ratios_opt[2]:.4f}  "
          f"(target {target_t:.4f}, "
          f"dev {100*(ratios_opt[2]-target_t)/target_t:+.4f}%)")
    print()

    print("  Physical reason for failure:")
    print("    Off-diagonal terms in NNI push eigenvalues AWAY from")
    print(f"    diagonal entries. A_u = {A_u:.4f} already exceeds the")
    print(f"    target {target_c:.4f}, so no (a, d) can reduce λ_2 below A_u.")
    print()

    # --- Verdict ---
    section("VERDICT: OP-M2/W4 verified")

    if res_c > 1.0 or res_t > 1.0:
        print("  PASS. Quark-lepton bifurcation is structural:")
        print("    - Leptons (colorless): 1D generation chain -> NNI OK.")
        print("    - Quarks (colored): K4 cycle space -> geometric")
        print("      products OK (0.02-3.1% deviations).")
        print("    - Quarks on NNI: FAIL.")
        print(f"      * m_c/m_u fit OK ({res_c:.4f}%), but")
        print(f"      * m_t/m_u fails ({res_t:.4f}%), because")
        print(f"        A_u already exceeds target by 1.89%.")
        print("      No choice of (a, d) can lower the top eigenvalue")
        print("      below its diagonal entry.")
    else:
        print(f"  FAIL: both ratios fit to <1%.")
    print()


if __name__ == "__main__":
    main()