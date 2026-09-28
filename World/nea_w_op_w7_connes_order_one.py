#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_w_op_w7_connes_order_one.py

OP-W7: Test whether the N.E.A. internal Dirac operator (M matrix)
satisfies the Connes order-one condition.

Connes spectral triple: (A, H, D)
Order-one condition: [[D, a], b] = 0 for all a, b in A

Test candidates:
  1. A_F = C^3 (diagonal algebra on 3 generations)
  2. A_F = C[2O] (group algebra of binary octahedral group)
  3. A_F = C + H + M_3(C) (Connes standard)
"""

import numpy as np
from itertools import permutations


# ── M matrix (from P-paper group theory) ──
def build_M():
    m_e = 0.51099895
    eps = 0.1
    a = 3.0 * eps
    A = 66.0 * np.pi
    d = 48.0
    B = 66.0 * np.pi * (16.0 / 3.0) * np.pi

    return m_e * np.array([
        [1.0, a, 0.0],
        [a, A, d],
        [0.0, d, B],
    ])


# ── Test 1: diagonal algebra C^3 ──
def test_diagonal_algebra(M):
    """
    A_F = C^3 acting as diag(a1, a2, a3) on the 3-gen space.
    Test [[M, a], b] = 0 for generic a, b.
    """
    # Generic diagonal elements
    a = np.diag([1.0, 2.0, 3.0])  # use explicit values, not symbolic
    b = np.diag([2.0, 5.0, 7.0])

    comm_Ma = M @ a - a @ M
    double_comm = comm_Ma @ b - b @ comm_Ma

    frob = np.linalg.norm(double_comm, 'fro')
    return frob, double_comm


# ── Test 2: diagonal algebra, general (a_j - a_i) structure ──
def test_diagonal_generic(M):
    """
    Analytically: [M, diag(a)]_(i,j) = M_ij * (a_j - a_i)
    So [[M, diag(a)], diag(b)]_(i,j) = M_ij (a_j - a_i)(b_j - b_i)

    Check whether this can be zero for all diagonal a, b.
    """
    # Enumerate: is there any non-trivial M satisfying this?
    # For M_ij != 0 and i != j, need (a_j - a_i)(b_j - b_i) = 0
    # for all a, b.
    # Counterexample: a = diag(0, 1, 2), b = diag(0, 1, 2)
    a = np.diag([0.0, 1.0, 2.0])
    b = np.diag([0.0, 1.0, 2.0])
    comm_Ma = M @ a - a @ M
    dc = comm_Ma @ b - b @ comm_Ma
    return np.linalg.norm(dc, 'fro'), dc


# ── Test 3: Connes standard A_F = C + H + M_3(C) ──
def test_connes_standard(M):
    """
    For 3 gen space alone, A_F = C^3 is the natural algebra.
    Connes standard A_F acts on larger 72-dim space, not on 3-gen alone.
    Here we test 3-gen restriction only.
    """
    # C acts as scalar * I_3
    # H (quaternions) acts as 2x2 on weak doublet (not on gen index)
    # M_3(C) acts on color index
    # None of these act on gen index within 3-gen space, except C scalar.

    # Test: C acts as scalar
    a = 2.0 * np.eye(3)
    b = 3.0 * np.eye(3)
    comm_Ma = M @ a - a @ M  # = 0 since a is scalar
    dc = comm_Ma @ b - b @ comm_Ma  # = 0
    return np.linalg.norm(dc, 'fro'), dc


# ── Test 4: general 3x3 matrix algebra M_3(C) ──
def test_M3C(M):
    """
    A_F = M_3(C) (all 3x3 complex matrices acting on gen space).
    Test order-one condition.
    """
    # Generic 3x3 matrices
    np.random.seed(42)
    a = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    b = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)

    comm_Ma = M @ a - a @ M
    dc = comm_Ma @ b - b @ comm_Ma
    return np.linalg.norm(dc, 'fro'), dc


# ── Main ──
def main():
    print("=" * 84)
    print("  OP-W7: Connes order-one condition for N.E.A. internal Dirac")
    print("=" * 84)
    print()

    M = build_M()
    print("M matrix:")
    print(M)
    print()

    # Test 1: diagonal algebra C^3 with specific element
    frob, _ = test_diagonal_algebra(M)
    print("[1] A_F = C^3 (diagonal algebra)")
    print("    ||[[M, a], b]||_F for specific a, b = {:.6e}".format(frob))
    print("    Order-one satisfied: {}".format(frob < 1e-10))
    print()

    # Test 2: diagonal algebra with structured elements
    frob, _ = test_diagonal_generic(M)
    print("[2] A_F = C^3 with (a_j - a_i) structure")
    print("    ||[[M, a], b]||_F = {:.6e}".format(frob))
    print("    Order-one satisfied: {}".format(frob < 1e-10))
    print("    Analytic: [M, diag(a)]_(i,j) = M_ij (a_j - a_i)")
    print("              [[M, diag(a)], diag(b)]_(i,j) = M_ij (a_j-a_i)(b_j-b_i)")
    print("    Nonzero whenever M_ij != 0 and a_j != a_i and b_j != b_i.")
    print()

    # Test 3: scalar algebra C
    frob, _ = test_connes_standard(M)
    print("[3] A_F = C (scalar algebra)")
    print("    ||[[M, a], b]||_F = {:.6e}".format(frob))
    print("    Order-one satisfied: {}".format(frob < 1e-10))
    print("    Trivially satisfied because scalars commute with M.")
    print()

    # Test 4: full matrix algebra M_3(C)
    frob, _ = test_M3C(M)
    print("[4] A_F = M_3(C) (full 3x3 complex matrix algebra)")
    print("    ||[[M, a], b]||_F = {:.6e}".format(frob))
    print("    Order-one satisfied: {}".format(frob < 1e-10))
    print()

    # Summary
    print("=" * 84)
    print("  Verdict")
    print("=" * 84)
    print()
    print("""
  For the 3-generation N.E.A. mass matrix M:

    A_F = C           -> order-one trivially satisfied (but trivial)
    A_F = C^3 (diag)  -> order-one FAILS
    A_F = M_3(C)      -> order-one FAILS

  This means: the N.E.A. internal Dirac operator does NOT fit
  the standard Connes order-one framework when A_F is non-scalar.

  Two interpretations:

  (A) N.E.A. uses a DIFFERENT spectral triple axiom set than Connes.
      The order-one condition is a technical constraint in Connes'
      reconstruction of the Standard Model, but is not physically
      necessary for a discrete spectral action.

  (B) The algebra A_F must be larger than C^3 on the gen space,
      acting on the FULL 72-dim space (gen * color * weak * spinor).
      The order-one condition may hold on the full 72-dim space
      even if it fails on the 3-gen subspace.

  Either way, the W paper Part VI must state this distinction
  explicitly: N.E.A. is structurally analogous to Connes NCG
  but does not use the same axiom set.

  This is a genuine finding, not a failure.
""")


if __name__ == "__main__":
    main()