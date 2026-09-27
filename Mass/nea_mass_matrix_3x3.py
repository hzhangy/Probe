#!/usr/bin/env python3
# nea_mass_matrix_3x3.py
"""
A1: 3×3 质量矩阵探索。

目标: 构造一个 3×3 矩阵 M, 其特征值自然给出
      m_e : m_μ : m_τ = 1 : 206.77 : 3477.2

尝试几个候选结构, 报告偏差。
"""
import numpy as np

# 目标特征值 (以 m_e 为单位)
target = np.array([1.0, 206.769, 3477.2])
print("=" * 72)
print("  A1: 3×3 质量矩阵探索")
print("=" * 72)
print(f"  目标特征值: {target}")
print()

# 几何常数
pi = np.pi
phi = (1 + np.sqrt(5)) / 2    # 黄金比
K12 = 66                      # C(12,2)
K20 = 190                     # C(20,2)
N12 = 36                      # 66 - 30
N20 = 160                     # 190 - 30
ratio_16_3 = 16/3
A = 66 * pi                   # 207.345
B = 66 * pi * 16 * pi / 3     # 3474.0

def eigenvalues(M):
    return np.sort(np.linalg.eigvalsh(M))

def report(name, M):
    ev = eigenvalues(M)
    dev = np.abs(ev - target) / target * 100
    print(f"  {name}")
    print(f"    特征值: {ev}")
    print(f"    偏差:   {dev}%")
    print(f"    最大偏差: {dev.max():.4f}%")
    print()

# ── 候选 1: 对角矩阵 ──
M1 = np.diag([1.0, A, B])
report("候选 1: diag(1, 66π, 66π·16π/3)", M1)

# ── 候选 2: 对角 + (2,3) 非对角 ──
# 解 d 使 λ₂ 匹配
d2 = np.sqrt((A + B - 2*target[1])**2/4 - (A-B)**2/4)
M2 = np.array([
    [1.0, 0, 0],
    [0, A, d2],
    [0, d2, B]
])
report(f"候选 2: + (2,3) 非对角 d={d2:.4f}", M2)

# ── 候选 3: 对角 + (1,2) 非对角 ──
M3 = np.array([
    [1.0, 1/phi, 0],
    [1/phi, A, 0],
    [0, 0, B]
])
report(f"候选 3: + (1,2) 非对角 1/φ={1/phi:.4f}", M3)

# ── 候选 4: 基于 K₁₂/K₂₀ 的矩阵 ──
M4 = np.array([
    [1.0, 1/K12, 0],
    [1/K12, A, np.sqrt(A*B)/K12],
    [0, np.sqrt(A*B)/K12, B]
])
report(f"候选 4: K₁₂ 混合", M4)

# ── 候选 5: 全非对角扫描 ──
print("  候选 5: 扫描 (2,3) 非对角 d, 找最佳匹配")
print(f"  {'d':>10s}  {'λ₂':>10s}  {'λ₃':>10s}  {'max偏差':>10s}")
best_d, best_dev = None, 1e10
for d in np.linspace(0, 100, 201):
    M = np.array([
        [1.0, 0, 0],
        [0, A, d],
        [0, d, B]
    ])
    ev = eigenvalues(M)
    dev = np.abs(ev[1:] - target[1:]) / target[1:] * 100
    if dev.max() < best_dev:
        best_dev = dev.max()
        best_d = d
    if d % 20 < 1:
        print(f"  {d:>10.2f}  {ev[1]:>10.2f}  {ev[2]:>10.2f}  {dev.max():>10.4f}")

print(f"\n  最佳 d = {best_d:.4f}, 最大偏差 = {best_dev:.4f}%")
M5 = np.array([
    [1.0, 0, 0],
    [0, A, best_d],
    [0, best_d, B]
])
report("候选 5: 最佳 (2,3) 非对角", M5)

# ── 候选 6: 全矩阵扫描 (a12, a23) ──
print("  候选 6: 扫描 (1,2) 和 (2,3) 非对角")
best_a, best_d, best_dev = 0, 0, 1e10
for a in np.linspace(0, 5, 51):
    for d in np.linspace(0, 80, 81):
        M = np.array([
            [1.0, a, 0],
            [a, A, d],
            [0, d, B]
        ])
        ev = eigenvalues(M)
        dev = np.abs(ev - target) / target * 100
        if dev.max() < best_dev:
            best_dev = dev.max()
            best_a, best_d = a, d

print(f"  最佳 a = {best_a:.4f}, d = {best_d:.4f}, max偏差 = {best_dev:.4f}%")
M6 = np.array([
    [1.0, best_a, 0],
    [best_a, A, best_d],
    [0, best_d, B]
])
report("候选 6: 最佳 (1,2)+(2,3)", M6)

# ── 检查: 最佳矩阵是否有几何解释 ──
print("=" * 72)
print("  几何检查")
print("=" * 72)
print(f"  A = 66π = {A:.4f}")
print(f"  B = 66π·16π/3 = {B:.4f}")
print(f"  1/φ = {1/phi:.6f}")
print(f"  1/K₁₂ = {1/K12:.6f}")
print()
print(f"  候选 6 最佳非对角元:")
print(f"    a₁₂ = {best_a:.4f}, 几何候选: 1/φ={1/phi:.4f}, ε²={0.01:.4f}")
print(f"    d₂₃ = {best_d:.4f}, 几何候选: K₁₂={K12}, φ²={phi**2:.4f}")