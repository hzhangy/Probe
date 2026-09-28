#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_w_gap1_a0_to_nnode.py

Gap 1: connect a0 = 4*pi (Part IV) to n_node (Paper D).

Strategy:
  Part IV:  a0/(4*pi) = 1, i.e. a0 = 4*pi.  a0 is the "node count per
            unit-sphere" in the continuous limit.
  Paper D:  n_node is the physical density in m^-3.

  Bridge:
    n_node = (node count per unit-sphere) * (unit-sphere -> physical volume)
           = a0 * (something with lambda_Z and N_max)

  Verify whether the topological genes (lambda_Z, N_max, Omega_Lambda,
  f_H, f_geo, R) reproduce Paper D's closed form.
"""

import numpy as np

# Constants
pi = np.pi
Delta = 1.0 - np.sqrt(3.0) / 2.0         # 0.133975
R = 1.0 / (1.0 + pi)                     # 0.241453
N_max = np.exp(10.0 * np.sqrt(3.0))      # 3.328e7
f_H = (19.0 - 4.0 * np.sqrt(3.0)) / 20.0 # 0.603590
f_geo = 1.0 + Delta / (4.0 * pi)         # 1.010660

# Physical
Z_MeV = 0.406640                          # MeV (Zhangyu, Paper F)
hbar_c_MeV_fm = 197.327
lambda_Z_fm = hbar_c_MeV_fm / Z_MeV      # 485.26 fm
lambda_Z_m = lambda_Z_fm * 1e-15         # m

# Cosmology (Paper D)
Omega_Lambda = 0.684527
Omega_m = 0.315473

# Paper D n_node
n_node_D = (1.0 / (lambda_Z_m**3 * N_max**4)) * (
    3.0 * Omega_Lambda * f_H**2 * f_geo**2
    / (8.0 * pi * (2.0 + pi)**2)
)
print("=" * 84)
print("  Gap 1: a0 -> n_node  bridge")
print("=" * 84)
print()
print("  --- Part IV side ---")
print("  a0/(4*pi) = 1.0  (dimensionless area)")
print("  a0 = 4*pi = {:.6f}".format(4 * pi))
print()
print("  --- Paper D side ---")
print("  n_node = {:.6e} m^-3".format(n_node_D))
print("  node spacing = {:.4f} cm".format((1.0 / n_node_D)**(1.0/3.0) * 100))
print()
print("  --- Bridge ---")
print()

# Attempt: n_node = a0 * (some topological factor) / (lambda_Z^3 * N_max^4)
#
# Paper D: n_node * lambda_Z^3 * N_max^4 = 3*Omega_L*f_H^2*f_geo^2 / (8*pi*(2+pi)^2)
#
# If a0 = 4*pi, then:
#   n_node * lambda_Z^3 * N_max^4 / a0 = 3*Omega_L*f_H^2*f_geo^2 / (32*pi^2*(2+pi)^2)

factor_D = 3.0 * Omega_Lambda * f_H**2 * f_geo**2 / (8.0 * pi * (2.0 + pi)**2)
factor_D_via_a0 = factor_D / (4.0 * pi)

print("  Paper D: n_node * lambda_Z^3 * N_max^4 = {:.6f}".format(factor_D))
print("  Via a0:  n_node * lambda_Z^3 * N_max^4 / a0 = {:.6f}".format(factor_D_via_a0))
print()

# Can factor_D_via_a0 be expressed via topological genes?
# Try: 3*Omega_L / (32*pi^2) * (f_H*f_geo/(2+pi))^2
term1 = 3.0 * Omega_Lambda / (32.0 * pi**2)
term2 = (f_H * f_geo / (2.0 + pi))**2
print("  Candidate decomposition:")
print("    term1 = 3*Omega_L / (32*pi^2)             = {:.6f}".format(term1))
print("    term2 = (f_H * f_geo / (2+pi))^2         = {:.6f}".format(term2))
print("    term1 * term2                              = {:.6f}".format(term1 * term2))
print("    factor_D_via_a0 (target)                   = {:.6f}".format(factor_D_via_a0))
print()

# Test: is there any combination of {Omega_L, f_H, f_geo, R, Delta} that
# gives factor_D_via_a0 exactly?
print("  --- Constraint from topology ---")
print()
# f_H = 19/20 - sqrt(3)/5 = (19 - 4*sqrt(3))/20
# f_H = 1 - Delta + ...  Let's check
Delta_related = 19.0/20.0 - 4.0*np.sqrt(3.0)/20.0
print("  f_H = (19 - 4*sqrt(3))/20")
print("  f_H = 1 - Delta + something?")
print("  f_H = {:.8f}".format(f_H))
print("  1 - Delta = {:.8f}".format(1.0 - Delta))
print("  f_H - (1 - Delta) = {:.8f}".format(f_H - (1.0 - Delta)))
print()

# f_geo = 1 + Delta/(4pi)
# Try: f_H * f_geo / (2 + pi)
ratio = f_H * f_geo / (2.0 + pi)
print("  f_H * f_geo / (2 + pi) = {:.8f}".format(ratio))
print("  R^2 = 1/(1+pi)^2 = {:.8f}".format(R**2))
print("  Delta / (1 + pi) = {:.8f}".format(Delta / (1.0 + pi)))
print()

# Test hypothesis: factor_D_via_a0 = 3*Omega_L * (something)
# Solve for something
something = factor_D_via_a0 / (3.0 * Omega_Lambda)
print("  Solve:  factor_D_via_a0 / (3*Omega_L) = {:.8f}".format(something))
print("  Compare to 1/(32*pi^2) = {:.8f}".format(1.0 / (32.0 * pi**2)))
print()

# Numerically check what n_node would be if factor = 3*Omega_L/(32*pi^2) * (f_H*f_geo/(2+pi))^2
n_node_recon = (1.0 / (lambda_Z_m**3 * N_max**4)) * 4.0 * pi * factor_D_via_a0
print("  --- Numerical reconstruction ---")
print("  n_node (reconstructed via a0) = {:.6e} m^-3".format(n_node_recon))
print("  n_node (Paper D directly)     = {:.6e} m^-3".format(n_node_D))
print("  Ratio                        = {:.10f}".format(n_node_recon / n_node_D))
print()

print("=" * 84)
print("  DIAGNOSIS")
print("=" * 84)
print()
print("""
  The a0 -> n_node bridge has THREE ingredients:
    (A) a0/(4pi) = 1               [geometric, Part IV]
    (B) 1/(lambda_Z^3 N_max^4)     [topological, Paper D]
    (C) 3*Omega_L f_H^2 f_geo^2 / (8pi(2+pi)^2)  [cosmological, Paper D]

  (A) and (B) are NATURAL from the framework.
  (C) is NOT a topological gene; it contains Omega_L (cosmological ratio)
      and the combination f_H^2 f_geo^2 / (8pi(2+pi)^2).

  QUESTION: Can (C) be expressed via topological genes ONLY?

  If YES:  a0 -> n_node is a genuine derivation.
  If NO:   a0 -> n_node is a NUMERICAL RELATION, not a theorem.

  Next: search for a topological formula for (C).
""")
print("=" * 84)