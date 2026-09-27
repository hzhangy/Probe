#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_multicenter_flrw_v5.py

OP-S1 v5: Particle-Mesh (PM) 埃瓦尔德方法。

v4 遗留: α = 0.183 (周期), 0.186 (有限球+金斯), 都偏离 1/6。
         怀疑是最小镜像截断 + 有限尺寸效应。

v5 方案: PM 方法在 FFT 网格上完整求解 ∇²φ = -Δ(ρ-ρ̄),
         无最小镜像截断, 可跑到 N = 10^6 ~ 10^7。
         这是标准宇宙学 N-body 的势场求解方法。
"""
import numpy as np


Delta = 1 - np.sqrt(3)/2
tau_0 = 100 * np.log(0.80/0.79)


def pm_phi(positions, L, N_grid, Delta):
    """PM 方法：CIC 分配 → FFT 求解 → CIC 插值。"""
    dx = L / N_grid
    pg = positions / dx
    i0 = np.floor(pg).astype(int) % N_grid
    i1 = (i0 + 1) % N_grid
    f = pg - np.floor(pg)
    fx, fy, fz = f[:,0], f[:,1], f[:,2]
    fx1, fy1, fz1 = 1-fx, 1-fy, 1-fz

    corners = [
        (i0[:,0], i0[:,1], i0[:,2], fx1*fy1*fz1),
        (i1[:,0], i0[:,1], i0[:,2], fx*fy1*fz1),
        (i0[:,0], i1[:,1], i0[:,2], fx1*fy*fz1),
        (i0[:,0], i0[:,1], i1[:,2], fx1*fy1*fz),
        (i1[:,0], i1[:,1], i0[:,2], fx*fy*fz1),
        (i1[:,0], i0[:,1], i1[:,2], fx*fy1*fz),
        (i0[:,0], i1[:,1], i1[:,2], fx1*fy*fz),
        (i1[:,0], i1[:,1], i1[:,2], fx*fy*fz),
    ]

    rho = np.zeros(N_grid**3)
    for ix, iy, iz, w in corners:
        idx = ix * N_grid**2 + iy * N_grid + iz
        rho += np.bincount(idx, weights=w, minlength=N_grid**3)
    rho = rho.reshape(N_grid, N_grid, N_grid)

    delta_rho = rho - rho.mean()
    delta_rho_k = np.fft.fftn(delta_rho)

    kx = np.fft.fftfreq(N_grid, d=dx) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[0,0,0] = 1.0

    phi_k = -Delta * delta_rho_k / k2
    phi_k[0,0,0] = 0.0
    phi_grid = np.real(np.fft.ifftn(phi_k))

    phi_i = np.zeros(len(positions))
    for ix, iy, iz, w in corners:
        phi_i += w * phi_grid[ix, iy, iz]

    return phi_i


def run_pm(N, n=1.0, N_grid=None, seed=42):
    rng = np.random.default_rng(seed)
    L = (N / n)**(1/3)
    pos = rng.random((N, 3)) * L
    if N_grid is None:
        N_grid = int(np.ceil((N/8)**(1/3) * 2))
        N_grid = max(N_grid, 16)
    phi = pm_phi(pos, L, N_grid, Delta)
    phi -= phi.mean()
    return N, L, N_grid, phi.std()


if __name__ == '__main__':
    print("=" * 78)
    print("  N.E.A. OP-S1 v5: Particle-Mesh 埃瓦尔德方法")
    print("=" * 78)
    print(f"  Delta = {Delta:.6f},  tau_0 = {tau_0:.6f}\n")

    print("─" * 78)
    print(f"  {'N':>10s}  {'L':>8s}  {'N_grid':>7s}  {'δφ_rms':>14s}")
    print("─" * 78)

    Ns, stds = [], []
    for N in [10**3, 10**4, 10**5, 10**6, 10**7]:
        Ni, L, Ng, s = run_pm(N)
        Ns.append(Ni); stds.append(s)
        print(f"  {Ni:10d}  {L:8.3f}  {Ng:7d}  {s:14.6f}")

    log_N = np.log(np.array(Ns))
    log_s = np.log(np.array(stds))

    print()
    print("─" * 78)
    print("  分段标度指数")
    print("─" * 78)
    for i in range(len(Ns)-1):
        sl = (log_s[i+1] - log_s[i]) / (log_N[i+1] - log_N[i])
        print(f"  N ∈ [{Ns[i]:.0e}, {Ns[i+1]:.0e}]:  α = {sl:.4f}")

    slope_all, _ = np.polyfit(log_N, log_s, 1)
    slope_late, _ = np.polyfit(log_N[-3:], log_s[-3:], 1)

    print()
    print("=" * 78)
    print(f"  全域拟合:     α = {slope_all:.4f}")
    print(f"  大 N 拟合:    α = {slope_late:.4f}")
    print(f"  理论渐近:     α = 1/6 = 0.1667")
    print("=" * 78)