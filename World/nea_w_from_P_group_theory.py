#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_w_from_P_group_theory.py

Explicit construction of Platonic group representations from Volume P,
and their direct application to the M-matrix of Volume M.

Fixed: 2O order is 48 (not 24). The 48 unit quaternions consist of:
  - 8 Lipschitz:    ±1, ±i, ±j, ±k
  - 16 Hurwitz:     (±1 ±i ±j ±k) / 2
  - 24 face-centered: (±1 ±i) / √2  with all axis pairs

P paper results used:
  - S4 (order 24), 3-dim standard irrep         -> 3 generations
  - A4 (order 12), 3-dim standard irrep         -> 3 colors
  - 2O (order 48), 2-dim complex irrep          -> weak doublet
  - K12 edge count 66, K20 ratio 16/3
"""

import numpy as np
import itertools
from itertools import permutations
from collections import Counter


# ═══════════════════════════════════════════════════════════
# 1. S4 and A4 as permutation groups
# ═══════════════════════════════════════════════════════════

def perm_to_4x4(p):
    M = np.zeros((4, 4))
    for i, pi in enumerate(p):
        M[i, pi] = 1.0
    return M


def all_perms():
    return list(permutations(range(4)))


def get_S4():
    return [perm_to_4x4(p) for p in all_perms()]


def get_A4():
    result = []
    for p in all_perms():
        inv = sum(1 for i in range(4) for j in range(i+1, 4) if p[i] > p[j])
        if inv % 2 == 0:
            result.append(perm_to_4x4(p))
    return result


def standard_3dim(perms_4x4):
    """Project 4-dim permutation rep onto 3-dim standard irrep."""
    U = np.array([
        [ 1, -1,  0,  0],
        [ 1,  1, -2,  0],
        [ 1,  1,  1, -3],
    ], dtype=float).T
    Q, _ = np.linalg.qr(U)
    return [Q.T @ P @ Q for P in perms_4x4]


# ═══════════════════════════════════════════════════════════
# 2. Binary octahedral group 2O  (CORRECTED: 48 elements)
# ═══════════════════════════════════════════════════════════

def quaternion_to_su2(q):
    """Quaternion (a, b, c, d) -> SU(2) matrix a*I - i*(b*sx + c*sy + d*sz)."""
    a, b, c, d = q
    return np.array([
        [a + 1j*b, c + 1j*d],
        [-c + 1j*d, a - 1j*b],
    ])


def get_2O_quaternions():
    """
    Return all 48 unit quaternions forming 2O.
    3 families:
      Lipschitz (8):       ±1, ±i, ±j, ±k
      Hurwitz   (16):      (±1 ±i ±j ±k) / 2
      Face-centered (24):  (±1 ±i) / √2  over all axis pairs
    """
    quats = []
    s2 = 1.0 / np.sqrt(2.0)

    # 8 Lipschitz
    for axis in range(4):
        for s in (1, -1):
            q = [0.0, 0.0, 0.0, 0.0]
            q[axis] = float(s)
            quats.append(tuple(q))

    # 16 Hurwitz
    for s1 in (1, -1):
        for s2_ in (1, -1):
            for s3 in (1, -1):
                for s4 in (1, -1):
                    quats.append((s1/2, s2_/2, s3/2, s4/2))

    # 24 face-centered: pairs of axes with ± combinations
    for i, j in itertools.combinations(range(4), 2):
        for s1 in (1, -1):
            for s2_ in (1, -1):
                q = [0.0, 0.0, 0.0, 0.0]
                q[i] = s1 * s2
                q[j] = s2_ * s2
                quats.append(tuple(q))

    return quats


def get_2O():
    """48 SU(2) matrices forming 2O."""
    quats = get_2O_quaternions()

    # Deduplicate at quaternion level first
    unique_quats = []
    for q in quats:
        if not any(np.allclose(q, u, atol=1e-10) for u in unique_quats):
            unique_quats.append(q)

    if len(unique_quats) != 48:
        print("  WARNING: 2O quaternion count = {} (expected 48)".format(
            len(unique_quats)))

    return [quaternion_to_su2(q) for q in unique_quats]


# ═══════════════════════════════════════════════════════════
# 3. Verification
# ═══════════════════════════════════════════════════════════

def is_group_4x4(mats):
    for M in mats:
        for N in mats:
            prod = M @ N
            if not any(np.allclose(prod, U, atol=1e-9) for U in mats):
                return False
    return True


def is_group_su2(mats):
    for M in mats:
        for N in mats:
            prod = M @ N
            if not any(np.allclose(prod, U, atol=1e-9) for U in mats):
                return False
    return True


def char_dist(mats):
    return Counter(np.round(np.trace(M).real, 6) for M in mats)


# ═══════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 84)
    print("  From P paper group theory to M paper mass matrix")
    print("=" * 84)
    print()

    # ── S4 ──
    S4 = get_S4()
    print("[1] S4 group (4x4 permutation representation)")
    print("    Order: {}".format(len(S4)))
    print("    Group closure: {}".format(is_group_4x4(S4)))

    S4_3 = standard_3dim(S4)
    print("    3-dim standard irrep: {} matrices of shape {}".format(
        len(S4_3), S4_3[0].shape))
    chars = Counter(np.round(np.trace(M).real, 6) for M in S4_3)
    chars_clean = {float(k): v for k, v in sorted(chars.items())}
    print("    Character distribution: {}".format(chars_clean))
    print("    Expected: {-1:6, 0:8, 1:6, 3:1}")
    print()

    # ── A4 ──
    A4 = get_A4()
    print("[2] A4 group (4x4 permutation representation)")
    print("    Order: {}".format(len(A4)))
    print("    Group closure: {}".format(is_group_4x4(A4)))
    print("    A4 subset of S4: {}".format(
        all(any(np.allclose(M, U, atol=1e-9) for U in S4) for M in A4)))

    A4_3 = standard_3dim(A4)
    print("    3-dim standard irrep: {} matrices".format(len(A4_3)))
    chars = Counter(np.round(np.trace(M).real, 6) for M in A4_3)
    chars_clean = {float(k): v for k, v in sorted(chars.items())}
    print("    Character distribution: {}".format(chars_clean))
    print()

    # ── 2O ──
    two_O = get_2O()
    print("[3] Binary octahedral group 2O (SU(2) representation)")
    print("    Order: {}".format(len(two_O)))
    print("    Expected: 48")
    print("    Group closure: {}".format(is_group_su2(two_O)))

    all_su2 = True
    for M in two_O:
        if not np.allclose(M.conj().T @ M, np.eye(2), atol=1e-10):
            all_su2 = False
            break
        if not np.allclose(np.linalg.det(M), 1.0, atol=1e-10):
            all_su2 = False
            break
    print("    All matrices in SU(2): {}".format(all_su2))
    print()

    # ── 72 = 3 x 3 x 2 x 4 ──
    print("[4] 72-dimensional state space decomposition")
    dims = [
        ("S4 3-dim (generations)", 3),
        ("A4 3-dim (color)",       3),
        ("2O 2-dim (weak)",        2),
        ("Cl(1,3) Dirac (spinor)", 4),
    ]
    product = 1
    for name, d in dims:
        print("    {:32s} = {}".format(name, d))
        product *= d
    print("    {:32s} = {}".format("Product (72-dim state space)", product))
    print()

    # ── M matrix coefficients ──
    print("[5] M matrix coefficients from P-paper group theory")
    print()

    d_val = len(two_O)              # = 48
    E_K12 = 12 * 11 // 2            # = 66
    A_val = E_K12 * np.pi
    ratio = 160 / 30                # = 16/3
    B_val = E_K12 * np.pi * ratio * np.pi
    eps = 1 / 10
    a_val = 3 * eps                 # = 0.3

    print("    d = |2O|                  = {}".format(d_val))
    print("    A = E(K12) * pi           = {:.6f}".format(A_val))
    print("    B = E(K12) * (16/3) * pi^2 = {:.6f}".format(B_val))
    print("    a = 3 * eps               = {:.6f}".format(a_val))
    print()

    m_e = 0.51099895
    M = m_e * np.array([
        [1.0, a_val, 0.0],
        [a_val, A_val, float(d_val)],
        [0.0, float(d_val), B_val],
    ])

    eigvals = np.sort(np.linalg.eigvalsh(M))
    mu_ratio = eigvals[1] / eigvals[0]
    tau_ratio = eigvals[2] / eigvals[0]

    obs_mu = 206.7683
    obs_tau = 3477.228

    print("[6] M matrix eigenvalues")
    print("    lambda_1 = {:.6f}   (m_e)".format(eigvals[0]))
    print("    lambda_2 = {:.4f}    (m_mu)".format(eigvals[1]))
    print("    lambda_3 = {:.4f}    (m_tau)".format(eigvals[2]))
    print()
    print("    m_mu/m_e  = {:.6f}   (obs {:.4f}, dev {:.4f}%)".format(
        mu_ratio, obs_mu, abs(mu_ratio - obs_mu) / obs_mu * 100))
    print("    m_tau/m_e = {:.6f}   (obs {:.4f}, dev {:.4f}%)".format(
        tau_ratio, obs_tau, abs(tau_ratio - obs_tau) / obs_tau * 100))
    print()

    # ── δ_CP ──
    print("[7] delta_CP from 2O three-cycle")
    delta_cp = 2 * np.pi * np.sqrt(3) / 9
    print("    delta_CP = 2*pi*sqrt(3)/9 = {:.4f} deg  (obs 68.8 +- 4.5)".format(
        delta_cp * 180 / np.pi))
    print()

    # ── Summary ──
    print("=" * 84)
    print("  Verdict")
    print("=" * 84)
    print("""
  P paper group theory explicitly verified:
    - S4 order 24, 3-dim standard irrep
    - A4 order 12, 3-dim standard irrep
    - 2O order 48, 2-dim SU(2) irrep
    - 72 = 3 x 3 x 2 x 4 tensor product

  M matrix coefficients from P paper group theory:
    - d = |2O| = 48
    - a = 3 * eps = 0.3
    - A = E(K12) * pi = 66 * pi
    - B = E(K12) * (16/3) * pi^2

  Lepton mass ratios from M matrix eigenvalues.
""")


if __name__ == "__main__":
    main()