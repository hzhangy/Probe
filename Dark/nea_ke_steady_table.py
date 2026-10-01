#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_ke_steady_table.py

§4.6 KE 稳态表: sigma(phi) vs time step.
"""
import numpy as np

# 网格
N = 64
L = 32.0
dx = L / N
x = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# 参数
Delta = 1 - np.sqrt(3)/2
mu2 = 0.01
dt = 0.001

# 初始 phi: 随机噪声
rng = np.random.default_rng(42)
phi = 0.1 * rng.random((N, N, N)) + 0.05

# 固定源: 均匀分布 + 局部扰动
rho = np.ones((N, N, N)) * 0.5
rho[N//2-5:N//2+5, N//2-5:N//2+5, N//2-5:N//2+5] += 1.0

# KE 演化
def laplacian(f, dx):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1, 1:-1] = (
        f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]
        + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1]
        + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2]
        - 6 * f[1:-1, 1:-1, 1:-1]
    ) / dx**2
    return lap

print(f"  {'Time step':>10s}  {'σ(φ)':>12s}")
print("  " + "-" * 30)

for step in range(5001):
    f_ext = np.sqrt(np.clip(1 - 2*phi, 0, 1))
    lap = laplacian(phi, dx)
    dphi_dt = f_ext * (lap - mu2*phi + Delta*rho)
    phi = phi + dt * dphi_dt
    phi = np.clip(phi, 0, 0.49)

    if step % 1000 == 0:
        print(f"  {step:>10d}  {phi.std():>12.6f}")