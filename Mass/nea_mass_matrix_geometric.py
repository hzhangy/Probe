#!/usr/bin/env python3
# nea_mass_matrix_geometric.py
"""A1 续: 非对角元的几何起源检验。"""
import numpy as np

pi = np.pi
phi = (1 + np.sqrt(5)) / 2
Delta = 1 - np.sqrt(3)/2
R = 1/(1 + pi)
eps = 0.1

# 群阶
S4, A4, O, O2 = 24, 12, 24, 48   # O2 = |2O|

# 图量
K4_V, K4_E = 4, 6
C8_V, C8_E = 8, 12
Oct_E = 12
Ico_E, Dod_E = 30, 30
K12_E, K20_E = 66, 190
N12, N20 = 36, 160

A = 66 * pi
B = 66 * pi * 16 * pi / 3
target = np.array([1.0, 206.769, 3477.2])

a_fit, d_fit = 0.3, 48.0

print("=" * 72)
print("  A1 续: 非对角元的几何起源")
print("=" * 72)
print(f"  拟合值: a = {a_fit}, d = {d_fit}")
print()

print("─" * 72)
print("  d = 48 候选")
print("─" * 72)
for name, val in [
    ("|2O|", O2),
    ("2|S4|", 2*S4),
    ("K4_V × C8_E", K4_V * C8_E),
    ("C8_V × K4_E", C8_V * K4_E),
    ("Oct_E × K4_V", Oct_E * K4_V),
    ("N12 + C8_E", N12 + C8_E),
    ("2 × 4!", 48),
]:
    dev = abs(val - d_fit) / d_fit * 100
    mark = " ←" if dev < 1 else ""
    print(f"  {name:<25s} = {val:>6.1f}  dev = {dev:>5.2f}%{mark}")
print()

print("─" * 72)
print("  a = 0.3 候选")
print("─" * 72)
for name, val in [
    ("3/10 = 代数/Stride", 3/10),
    ("|2O|/N20", O2/N20),
    ("Δ + Δ²", Delta + Delta**2),
    ("Δ² + Δ³ + Δ⁴", Delta**2 + Delta**3 + Delta**4),
    ("R/φ", R/phi),
    ("ε×φ", eps*phi),
    ("(φ-1)/2", (phi-1)/2),
    ("1/π", 1/pi),
    ("Δ×(1+Δ)", Delta*(1+Delta)),
    ("1/(2π) + Δ/2", 1/(2*pi) + Delta/2),
    ("A4/(2|S4|-K4_E)", A4/(2*S4 - K4_E)),
    ("S4/|2O| + Δ", S4/O2 + Delta),
]:
    dev = abs(val - a_fit) / a_fit * 100
    mark = " ←" if dev < 1 else ""
    print(f"  {name:<25s} = {val:>10.6f}  dev = {dev:>7.2f}%{mark}")
print()

print("─" * 72)
print("  d 与 √(A×B) 的关系")
print("─" * 72)
sqrt_AB = np.sqrt(A * B)
ratio = sqrt_AB / d_fit
print(f"  √(A × B) = {sqrt_AB:.4f}")
print(f"  √(A×B) / d = {ratio:.4f}")
print()
for name, val in [
    ("N20/A4", N20/A4),
    ("|2O|", O2),
    ("(2+π)²", (2+pi)**2),
    ("φ² × |2O|", phi**2 * O2),
    ("K12_E", K12_E),
    ("2 × φ³", 2 * phi**3),
    ("π × |2O|", pi * O2),
    ("A4 × |2O| / 10", A4 * O2 / 10),
]:
    dev = abs(val - ratio) / ratio * 100
    mark = " ←" if dev < 2 else ""
    print(f"    {name:<22s} = {val:>10.4f}  dev = {dev:>7.2f}%{mark}")
print()

print("─" * 72)
print("  几何矩阵检验")
print("─" * 72)
a_geo = O2 / N20
d_geo = O2

M = np.array([
    [1.0, a_geo, 0],
    [a_geo, A, d_geo],
    [0, d_geo, B]
])
ev = np.sort(np.linalg.eigvalsh(M))
dev = np.abs(ev - target) / target * 100

print(f"  a = |2O|/N₂₀ = {O2}/{N20} = {a_geo}")
print(f"  d = |2O| = {d_geo}")
print()
print(f"  特征值: {ev}")
print(f"  偏差: {dev}%")
print(f"  最大偏差: {dev.max():.4f}%")
print()

print("─" * 72)
print("  物理图像")
print("─" * 72)
print()
print(f"  |2O| = {O2} = 二元八面体群的阶 (弱力载体)")
print(f"  N₂₀ = {N20} = E(K₂₀) - E(十二面体) (τ子模板)")
print()
print(f"  a = |2O|/N₂₀ = {a_geo}  (混合强度)")
print(f"  d = |2O| = {d_geo}  (μ-τ耦合)")
print()
print("  对角项: A = 66π, B = 66π·16π/3 (已有几何解释)")
print("  非对角项: a, d 由 2O 群阶与 K₂₀ 模板决定")