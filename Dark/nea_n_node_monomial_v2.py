#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nea_n_node_monomial_v2.py — 修正 c² + 检验 N_max^-4"""
import numpy as np

Delta = 1 - np.sqrt(3)/2
U_EM = 0.4*np.pi
U_weak = 10*np.sqrt(3)
N_max = np.exp(U_weak)
R = 1/(1+np.pi)
Omega_L = 1/(1 + (U_EM - 1/U_EM))
f_H = (19 - 4*np.sqrt(3))/20
f_geo = 1 + Delta/(4*np.pi)

lambda_Z = (197.3269804/0.406640)*1e-15
C_topo = 3*Omega_L*f_H**2*f_geo**2 / (8*np.pi*(2+np.pi)**2)

# 微观
n_topo = (1/lambda_Z**3) * (1/N_max**4) * C_topo
d_topo_cm = (1/n_topo)**(1/3) * 100

# 宏观 (修正 c²)
H0 = 68.04e3 / 3.08567758149e22  # s^-1
G = 6.6743e-11
c = 2.99792458e8
Z_J = 0.406640 * 1.602176634e-13
n_macro = 3*Omega_L*H0**2*c**2 / (8*np.pi*G*Z_J)
d_macro_cm = (1/n_macro)**(1/3) * 100

print(f"  微观: n = {n_topo:.4e} m^-3, d = {d_topo_cm:.4f} cm")
print(f"  宏观: n = {n_macro:.4e} m^-3, d = {d_macro_cm:.4f} cm")
print(f"  偏差: {abs(n_topo-n_macro)/n_macro*100:.3f}%")