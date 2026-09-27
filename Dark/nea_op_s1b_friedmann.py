#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_op_s1b_friedmann.py

OP-S1b: 从多中心稳态导出 Friedmann 方程。

策略:
  1. 在周期盒子中, 定义宏观标度因子 a = <r>/r_max
  2. 测量 a 随 KE 演化时间 t_KE 的变化
  3. 提取 H = (1/a)(da/dt_KE)
  4. 对比 H^2 与 rho = n * E_trap
"""
import numpy as np

Delta = 1 - np.sqrt(3)/2
tau_0 = 100 * np.log(0.80/0.79)
E_trap = 1.0  # 单位: ZY, 每个 K4 缺陷的被困租金


def ke_step(phi, dx, dt, mu2=0.0, sigma=0.0):
    lap = np.zeros_like(phi)
    lap[1:-1,1:-1,1:-1] = (
        phi[2:,1:-1,1:-1] + phi[:-2,1:-1,1:-1] +
        phi[1:-1,2:,1:-1] + phi[1:-1,:-2,1:-1] +
        phi[1:-1,1:-1,2:] + phi[1:-1,1:-1,:-2] -
        6*phi[1:-1,1:-1,1:-1]
    ) / dx**2
    rhs = lap - mu2*phi + Delta*sigma
    f_ext = np.sqrt(np.clip(1 - 2*phi, 0.0, 1.0))
    return phi + dt * f_ext * rhs


def initialize_multicentre(N, L, Delta, seed=42):
    rng = np.random.default_rng(seed)
    pos = rng.random((N, 3)) * L
    N_grid = max(int(np.ceil((N/8)**(1/3) * 2)), 16)
    dx = L / N_grid
    pg = pos / dx
    i0 = np.floor(pg).astype(int) % N_grid
    i1 = (i0 + 1) % N_grid
    f = pg - np.floor(pg)
    fx, fy, fz = f[:,0], f[:,1], f[:,2]
    fx1, fy1, fz1 = 1-fx, 1-fy, 1-fz
    corners = [
        (i0[:,0],i0[:,1],i0[:,2],fx1*fy1*fz1),
        (i1[:,0],i0[:,1],i0[:,2],fx*fy1*fz1),
        (i0[:,0],i1[:,1],i0[:,2],fx1*fy*fz1),
        (i0[:,0],i0[:,1],i1[:,2],fx1*fy1*fz),
        (i1[:,0],i1[:,1],i0[:,2],fx*fy*fz1),
        (i1[:,0],i0[:,1],i1[:,2],fx*fy1*fz),
        (i0[:,0],i1[:,1],i1[:,2],fx1*fy*fz),
        (i1[:,0],i1[:,1],i1[:,2],fx*fy*fz),
    ]
    rho = np.zeros((N_grid, N_grid, N_grid))
    for ix,iy,iz,w in corners:
        np.add.at(rho, (ix,iy,iz), w)
    rho -= rho.mean()
    rho_k = np.fft.fftn(rho)
    kx = np.fft.fftfreq(N_grid, d=dx) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[0,0,0] = 1.0
    phi_k = -Delta * rho_k / k2
    phi_k[0,0,0] = 0.0
    phi_grid = np.real(np.fft.ifftn(phi_k))
    return phi_grid, pos, dx, N_grid


def measure_scale_factor(phi, N_grid, L):
    """从 phi 场测量有效标度因子 a 与涨落幅度。"""
    x = (np.arange(N_grid) - N_grid/2) * (L/N_grid)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_max = L * np.sqrt(3) / 2
    # 加权半径
    weight = phi**2
    if weight.sum() > 0:
        r_avg = np.sum(r * weight) / weight.sum()
    else:
        r_avg = r_max / 2
    a = r_avg / r_max
    return a, r_avg, r_max


if __name__ == '__main__':
    print("=" * 78)
    print("  N.E.A. OP-S1b: 从多中心稳态导出 Friedmann 方程")
    print("=" * 78)
    print(f"  Delta = {Delta:.6f},  tau_0 = {tau_0:.6f},  E_trap = {E_trap}\n")

    N = 10000
    L = N**(1/3)
    n_density = N / L**3
    phi, pos, dx, N_grid = initialize_multicentre(N, L, Delta)

    print(f"  N = {N}, L = {L:.4f}, n = {n_density:.4f}")
    print(f"  N_grid = {N_grid}, dx = {dx:.4f}")
    print(f"  rho_expected = n * E_trap = {n_density * E_trap:.6f}\n")

    # 演化 KE, 测量 a(t)
    print("─" * 78)
    print(f"  {'step':>6s}  {'t_KE':>10s}  {'a':>10s}  {'da/dt':>12s}  "
          f"{'H':>12s}  {'H^2':>12s}  {'8piG/3 * rho':>14s}")
    print("─" * 78)

    dt = 0.005
    history = []
    phi_curr = phi.copy()

    for step_target in [0, 500, 1000, 2000, 5000, 10000, 20000]:
        if step_target > 0:
            n_iter = step_target - history[-1][0] if history else step_target
            for _ in range(n_iter):
                phi_curr = ke_step(phi_curr, dx, dt)
        t_KE = step_target * dt
        a, r_avg, r_max = measure_scale_factor(phi_curr, N_grid, L)
        history.append((step_target, t_KE, a))
        print(f"  {step_target:6d}  {t_KE:10.4f}  {a:10.6f}  "
              f"{'---':>12s}  {'---':>12s}  {'---':>12s}  "
              f"{n_density * E_trap:14.6f}")

    # 从历史中算 H
    if len(history) >= 2:
        print()
        print("─" * 78)
        print("  膨胀率 H = (1/a)(da/dt) 与 Friedmann 对比")
        print("─" * 78)
        print(f"  {'区间':>14s}  {'H':>14s}  {'H^2':>14s}  {'8piG/3*rho':>14s}  "
              f"{'H^2/(8piG/3*rho)':>18s}")
        print("─" * 78)
        # 用 G_eff = Delta / (4pi) (从 G 卷)
        G_eff = Delta / (4 * np.pi)
        friedmann_target = (8 * np.pi * G_eff / 3) * n_density * E_trap
        for i in range(len(history) - 1):
            s0, t0, a0 = history[i]
            s1, t1, a1 = history[i+1]
            if t1 > t0 and a0 > 0:
                H = (a1 - a0) / (a0 * (t1 - t0))
                ratio = H**2 / friedmann_target if friedmann_target > 0 else 0
                print(f"  [{s0:5d},{s1:5d}]  {H:14.6f}  {H**2:14.6f}  "
                      f"{friedmann_target:14.6f}  {ratio:18.4f}")

    print()
    print("=" * 78)
    print("  [物理解读]")
    print("=" * 78)
    print(f"""
  1. 多中心稳态的 KE 演化: 标度因子 a 从 {history[0][2]:.4f} 演化到 {history[-1][2]:.4f}。

  2. 膨胀率 H 从有限差分提取。

  3. Friedmann 目标值 8πG/3 * rho = {friedmann_target:.6f}, 
     其中 G_eff = Δ/(4π) = {G_eff:.6f}。

  4. 如果 H^2 / target -> 1, 则 Friedmann 方程从稳态导出。

  5. 注意: 这里用的是 KE 时间 t_KE, 不是 FLRW 时间 t_FLRW。
     两者的映射需要额外步骤 (OP-S1b 第二步)。
    """)
    print("=" * 78)