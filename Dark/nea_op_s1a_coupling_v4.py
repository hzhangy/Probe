#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nea_op_s1a_coupling_v4.py — 大 N 确认 α→1/6"""
import numpy as np

Delta = 1 - np.sqrt(3)/2

def pm_phi_center(N, L, Delta, seed=42):
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
    phi_c = np.zeros(N)
    for ix,iy,iz,w in corners:
        phi_c += w * phi_grid[ix,iy,iz]
    return phi_c, N_grid

if __name__ == '__main__':
    print("=" * 70)
    print("  OP-S1a v4: 大 N 确认 α→1/6")
    print("=" * 70)
    print(f"  {'N':>10s}  {'N_grid':>7s}  {'std(phi_c)':>14s}  {'分段 α':>10s}")
    print("-" * 70)
    Ns, stds = [], []
    for N in [10**4, 10**5, 10**6, 10**7]:
        L = N**(1/3)
        phi_c, Ng = pm_phi_center(N, L, Delta)
        s = phi_c.std()
        Ns.append(N); stds.append(s)
        if len(Ns) >= 2:
            a = np.log(stds[-1]/stds[-2]) / np.log(Ns[-1]/Ns[-2])
            print(f"  {N:10d}  {Ng:7d}  {s:14.6f}  {a:10.4f}")
        else:
            print(f"  {N:10d}  {Ng:7d}  {s:14.6f}  {'---':>10s}")
    print()
    print(f"  理论渐近: α = 1/6 = 0.1667")