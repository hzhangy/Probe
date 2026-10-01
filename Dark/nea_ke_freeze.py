#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_ke_freeze_v6.py

正确设计: 长时间演化, 让 φ 场有机会响应。
"""
import numpy as np

print("=" * 70)
print("  KE 冻结: 长时间演化")
print("=" * 70)
print()

N = 600
L = 400.0
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

Delta = 0.134
mu2 = 0.01
dt = 0.0001
v = 0.5  # 慢速

sigma_star = 3.0
sigma_gas = 8.0
x_star0 = -80.0
x_gas0 = -40.0

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma)**2)

def rho_at(x, t):
    x_s = x_star0 + v * t
    x_g = x_gas0 + v * t
    rho_s = 3.0 * gaussian(x, x_s, sigma_star)
    rho_g = 0.3 * gaussian(x, x_g, sigma_gas)
    return rho_s, rho_g

# 初始 phi: 让源在 t=0 达到稳态
# 用稳态解 phi_steady ~ Delta * rho / mu2 (大 mu2 时)
# 简化: 用源形状
phi = 0.48 * gaussian(x, x_star0, sigma_star) + \
      0.10 * gaussian(x, x_gas0, sigma_gas)
print(f"  初始 φ_max = {phi.max():.4f}")
print()

def find_peak(phi, x, x0, w=30):
    m = np.abs(x - x0) < w
    if m.sum() > 0:
        i = np.argmax(phi[m])
        return x[m][i], phi[m][i]
    return np.nan, np.nan

steps = 20000
rec = 2000
t = 0

print(f"  {'t':>6s}  {'φ* pos':>9s}  {'φ* exp':>9s}  {'lag*':>7s}  {'φ*':>7s}  "
      f"{'φg pos':>9s}  {'φg exp':>9s}  {'lag_g':>7s}  {'φ_g':>7s}")
print("  " + "-" * 88)

for step in range(steps):
    rho_s, rho_g = rho_at(x, t)
    rho = rho_s + rho_g
    
    lap = np.zeros(N)
    lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
    
    grad = np.gradient(phi, dx)
    
    f_ext = np.sqrt(np.clip(1 - 2*phi, 0, 1))
    
    convection = - f_ext * v * grad
    dphi_dt = f_ext * (lap - mu2 * phi + Delta * rho) + convection
    
    phi = phi + dt * dphi_dt
    phi = np.clip(phi, 0, 0.499)
    
    t += dt
    
    if step % rec == 0:
        x_s_exp = x_star0 + v * t
        x_g_exp = x_gas0 + v * t
        
        xs_p, phi_s = find_peak(phi, x, x_s_exp, 30)
        xg_p, phi_g = find_peak(phi, x, x_g_exp, 30)
        
        lag_s = x_s_exp - xs_p if not np.isnan(xs_p) else np.nan
        lag_g = x_g_exp - xg_p if not np.isnan(xg_p) else np.nan
        
        print(f"  {t:>6.2f}  {xs_p:>9.2f}  {x_s_exp:>9.2f}  {lag_s:>7.2f}  "
              f"{phi_s:>7.4f}  {xg_p:>9.2f}  {x_g_exp:>9.2f}  {lag_g:>7.2f}  "
              f"{phi_g:>7.4f}")

print()
print("=" * 70)
print("  正确判据:")
print("  星系 f_ext → 0, 对流系数 → 0, lag* 应该 >> lag_g")
print("  气体 f_ext ≈ 1, 对流正常, lag_g 应该小")
print("=" * 70)