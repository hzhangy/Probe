#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_mass_Ih_matrix.py

从二十面体对称群 I_h 的 3 维不可约表示构造带电轻子质量矩阵。

思路:
  二十面体旋转群 I = A_5 有 3 维不可约表示。
  质量矩阵 M 作用在世代空间 (3 维) 上。
  如果 M 不是 I 等变的, 可以给出非平凡本征值。
"""
import numpy as np
from numpy.linalg import eigvalsh

print("=" * 70)
print("  I_h 表示论: 3×3 质量矩阵构造")
print("=" * 70)
print()

# ── 目标 ──
m_e = 1.0
m_mu = 206.768
m_tau = 3477.15

print(f"  目标: {{1, {m_mu:.3f}, {m_tau:.3f}}}")
print()

# ── 二十面体顶点 ──
phi = (1 + np.sqrt(5)) / 2
V = np.array([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
])
V = V / np.linalg.norm(V[0])

# ── 构造 I 在 3D 中的生成元 ──
# 二十面体旋转群由以下生成:
# - 绕穿过对径顶点的轴旋转 72° (5 阶)
# - 绕穿过对径边中点的轴旋转 180° (2 阶)

# 选一条穿过顶点的轴
axis1 = V[0] / np.linalg.norm(V[0])
theta1 = 2 * np.pi / 5

def rot(axis, theta):
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K

R5 = rot(axis1, theta1)

# 找一条穿过边中点的轴
edge_mid = (V[0] + V[1]) / 2
axis2 = edge_mid / np.linalg.norm(edge_mid)
R2 = rot(axis2, np.pi)

print("  A_5 生成元:")
print(f"    R5 (5 阶): 迹 = {np.trace(R5):.6f}")
print(f"    R2 (2 阶): 迹 = {np.trace(R2):.6f}")
print(f"    R5^5 与 I 的差: {np.max(np.abs(np.linalg.matrix_power(R5,5) - np.eye(3))):.2e}")
print(f"    R2^2 与 I 的差: {np.max(np.abs(np.linalg.matrix_power(R2,2) - np.eye(3))):.2e}")
print(f"    (R5 R2)^3 与 I 的差: {np.max(np.abs(np.linalg.matrix_power(R5@R2,3) - np.eye(3))):.2e}")
print()

# ── 尝试 1: M = a I + b(R5 + R5^T) + c(R5^2 + R5^T^2) ──
print("─" * 70)
print("  尝试 1: R5 的多项式 (实对称)")
print("─" * 70)

R5T = R5.T
R5sq = R5 @ R5
R5sqT = R5sq.T

M1 = 1.0 * np.eye(3) + 10.0 * (R5 + R5T) + 5.0 * (R5sq + R5sqT)
eigs1 = np.sort(eigvalsh(M1))
print(f"  本征值: {eigs1}")
print(f"  比值: {eigs1[1]/eigs1[0]:.4f}, {eigs1[2]/eigs1[0]:.4f}")
print()

# ── 尝试 2: 用 R2 和 R5 的组合 ──
print("─" * 70)
print("  尝试 2: R5, R2 的组合")
print("─" * 70)

# 扫描参数, 寻找本征值比
best_chi2 = np.inf
best_params = None

for a in [1, 5, 10, 50, 100]:
    for b in [-10, -1, 0, 1, 10]:
        for c in [-10, -1, 0, 1, 10]:
            M = a*np.eye(3) + b*(R5 + R5T) + c*(R2 + R2.T)
            eigs = np.sort(np.abs(eigvalsh(M)))
            if eigs[0] < 1e-10:
                continue
            # 比值
            r1 = eigs[1]/eigs[0]
            r2 = eigs[2]/eigs[0]
            chi2 = (r1 - m_mu/m_e)**2 / (m_mu/m_e)**2 + (r2 - m_tau/m_e)**2 / (m_tau/m_e)**2
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_params = (a, b, c, eigs, r1, r2)

if best_params:
    a, b, c, eigs, r1, r2 = best_params
    print(f"  最佳: a={a}, b={b}, c={c}")
    print(f"  本征值: {eigs}")
    print(f"  比值: {r1:.4f}, {r2:.4f}")
    print(f"  目标: {m_mu/m_e:.4f}, {m_tau/m_e:.4f}")
    print(f"  χ² = {best_chi2:.4f}")
print()

# ── 尝试 3: 从二十面体图的 3 维特征空间 ──
print("─" * 70)
print("  尝试 3: 二十面体图 Laplacian 的 3 维特征空间")
print("─" * 70)

# 构造二十面体图 Laplacian
N = 12
L = np.zeros((N, N))
a_edge = np.linalg.norm(V[0] - V[1])
for i in range(N):
    for j in range(i+1, N):
        if abs(np.linalg.norm(V[i] - V[j]) - a_edge) < 1e-6:
            L[i, i] += 1
            L[j, j] += 1
            L[i, j] -= 1
            L[j, i] -= 1

eigs_L, evecs_L = np.linalg.eigh(L)
print(f"  Laplacian 本征值: {np.sort(eigs_L)}")
print()

# 找 3 维特征空间
target = 5 - np.sqrt(5)
idx_3d = np.where(np.abs(eigs_L - target) < 1e-6)[0]
print(f"  {target:.6f} 的特征空间维数: {len(idx_3d)}")
print()

# 该特征空间基底 (12×3)
subspace = evecs_L[:, idx_3d]

# 在 3 维空间中, 作用某些自然 12×12 矩阵
# 例如: 顶点坐标矩阵 M_full[i,j] = V_i · V_j

M_full = V @ V.T  # 12×12
M_3 = subspace.T @ M_full @ subspace

eigs_M = np.sort(np.abs(eigvalsh(M_3)))
print(f"  M_3 本征值: {eigs_M}")
print()

# ── 尝试 4: 更复杂的构造 ──
print("─" * 70)
print("  尝试 4: 二十面体轴矩阵")
print("─" * 70)

# 6 条对径顶点轴
axes = []
for i in range(6):
    axes.append(V[i] / np.linalg.norm(V[i]))
axes = np.array(axes)

# 矩阵 M = Σ_i w_i (a_i a_i^T)
# 尝试不同权重

# 尝试: w_i = (a_i · z)^n for some n
z = np.array([1, 0, 0])

for n in [1, 2, 3, 4]:
    weights = (axes @ z) ** n
    M4 = np.zeros((3, 3))
    for i in range(6):
        M4 += weights[i] * np.outer(axes[i], axes[i])
    eigs4 = np.sort(np.abs(eigvalsh(M4)))
    if eigs4[0] > 1e-10:
        print(f"  n={n}: 权重={weights}, eigs={eigs4}, "
              f"比值={eigs4[1]/eigs4[0]:.3f}, {eigs4[2]/eigs4[0]:.3f}")
print()

# ── 尝试 5: 二十面体 3 维表示的显式生成元 ──
print("─" * 70)
print("  尝试 5: 用二十面体旋转矩阵本身")
print("─" * 70)

# 构造两个旋转矩阵的特定组合
# 思路: 用 R5 的实部 (特征值 1, cos 2π/5, cos 4π/5)

# R5 的特征多项式
# det(R5 - λI) = 0
# λ = 1, exp(±2πi/5)

# R5 的实特征分解
# 实特征值 1, 实 2 维旋转子空间

# 矩阵可以按特征值展开
# M = I + c1 (R5 - <R5>) + c2 (R5 - <R5>)^2

# 用 (R5 - R5^T) 是反对称的
print(f"  R5 - R5^T 是反对称: max|A + A^T| = "
      f"{np.max(np.abs(R5 - R5.T + (R5 - R5.T).T)):.2e}")

# 用对称部分
M5 = np.eye(3) * 1.0 + 100.0 * (R5 + R5.T) / 2 + 50.0 * (R5 @ R5 + R5.T @ R5.T) / 2
eigs5 = np.sort(eigvalsh(M5))
print(f"  M5 本征值: {eigs5}")
print(f"  比值: {eigs5[1]/eigs5[0]:.4f}, {eigs5[2]/eigs5[0]:.4f}")
print()

print("=" * 70)
print("  结论")
print("=" * 70)
print("""
  基于 I_h 表示论的简单构造都失败。
  
  根本原因: I_h 的 3 维不可约表示上的等变矩阵只能是标量 (Schur 引理)。
  非等变矩阵需要破坏对称性, 但破坏方式由什么决定?
  
  真正的质量矩阵可能:
  1. 来自更大的空间 (72 维) 的投影
  2. 涉及 I_h 的两个 3 维不可约表示 (3 和 3')
  3. 涉及非平凡的复相位
  
  这是 months 级的工作, 不是对话级的尝试。
""")