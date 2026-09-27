#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_op_d4_heat_kernel_v2.py

OP-D4: 离散热核 vs 连续热核 (谱展开版)。
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

def build_periodic_cubic_sparse(L):
    """稀疏周期立方体拉普拉斯。"""
    N = L**3
    A = lil_matrix((N, N))
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = x * L**2 + y * L + z
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx = (x+dx) % L
                    ny = (y+dy) % L
                    nz = (z+dz) % L
                    j = nx * L**2 + ny * L + nz
                    A[i, j] = 1.0
    degrees = np.array(A.sum(axis=1)).flatten()
    D = lil_matrix((N, N))
    for i in range(N):
        D[i, i] = degrees[i]
    return (D - A).tocsr()

def heat_kernel_cont(t, r, d=3):
    """连续热核 (d 维)"""
    return (4 * np.pi * t)**(-d/2) * np.exp(-r**2 / (4*t))

print("=" * 78)
print("  OP-D4: 离散热核 vs 连续热核 (谱展开)")
print("=" * 78)
print()

L = 16
N = L**3
print(f"  L = {L}, N = {N}")

L_op = build_periodic_cubic_sparse(L)

# 计算最低 k 个本征值
k_eigs = 500
print(f"  计算前 {k_eigs} 个本征值...")
eigs, vecs = eigsh(L_op, k=k_eigs, which='SM', maxiter=10000)
eigs = np.sort(eigs)
idx = np.argsort(eigs)
eigs = eigs[idx]
vecs = vecs[:, idx]
print(f"  本征值范围: {eigs[0]:.6f} 到 {eigs[-1]:.4f}")
print()

# 原点索引
origin = 0

print(f"  {'t':>6s}  {'r':>4s}  {'K_disc':>14s}  {'K_cont':>14s}  {'比值':>10s}")
print("  " + "-" * 54)

for t in [0.5, 1.0, 2.0, 5.0]:
    for r_target in [1, 2, 3, 5]:
        if r_target >= L:
            continue
        j = r_target * L**2
        # 谱展开: K_t(0,j) = Σ_k e^{-t λ_k} ψ_k(0) ψ_k(j)
        K_d = np.sum(np.exp(-t * eigs) * vecs[origin, :] * vecs[j, :])
        K_c = heat_kernel_cont(t, r_target)
        ratio = K_d / K_c if K_c > 1e-30 else np.nan
        print(f"  {t:6.2f}  {r_target:4d}  {K_d:14.8f}  {K_c:14.8f}  {ratio:10.4f}")
    print()

print("=" * 78)
print("  [结论]")
print("=" * 78)
print("""
  用谱展开代替 expm, 避免矩阵指数溢出。
  
  离散热核 K_t(i,j) = Σ_k e^{-tλ_k} ψ_k(i) ψ_k(j)
  连续热核 K_t^cont(x,y) = (4πt)^{-3/2} exp(-|x-y|²/(4t))
  
  预期: t 大时 (长时极限) 比值趋近 1。
""")
print("=" * 78)