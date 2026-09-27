#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nea_delta_alpha_topology.py"""
import numpy as np

Delta = 1 - np.sqrt(3)/2
R = 1/(1 + np.pi)
eps = 0.1
N_max = np.exp(10*np.sqrt(3))
B1_octa = 7
B1_C8 = 5
B1_K4 = 3

alpha_0_inv = 137.035999
alpha_MZ_inv = 127.90
delta_alpha = 1/alpha_MZ_inv - 1/alpha_0_inv
target = delta_alpha * alpha_0_inv

print("=" * 70)
print("  Δα 拓扑结构探索")
print("=" * 70)
print(f"\nΔα (标准物理) = {delta_alpha:.6f}")
print(f"Δα/α(0) = {target:.6f}")
print()

candidates = {
    "(B1_octa) * eps^2 / 2":              B1_octa * eps**2 / 2,
    "(B1_octa + B1_C8) * eps^2 / 2":      (B1_octa + B1_C8) * eps**2 / 2,
    "(B1_octa+B1_C8+B1_K4) * eps^2 / 2":  (B1_octa + B1_C8 + B1_K4) * eps**2 / 2,
    "B1_octa * eps^2":                     B1_octa * eps**2,
    "(sum B1) * eps^2 / 2":                (B1_K4 + B1_C8 + B1_octa) * eps**2 / 2,
    "ln(N_max) * eps^2":                   np.log(N_max) * eps**2,
    "B1_octa * ln(N_max) * eps^2":         B1_octa * np.log(N_max) * eps**2,
    "(1/Delta) * eps^2":                   (1/Delta) * eps**2,
    "B1_octa * eps^2 / Delta":             B1_octa * eps**2 / Delta,
}

print(f"{'候选':>40s}  {'值':>12s}  {'比值':>12s}")
print("-" * 70)
for name, val in candidates.items():
    ratio = val / target
    print(f"{name:>40s}  {val:12.6f}  {ratio:12.4f}")

print()
print(f"目标值 = {target:.6f}")