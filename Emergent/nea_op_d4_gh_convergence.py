#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_op_d4_gh_convergence.py

OP-D4 第三步: 图距离 vs 连续测地距离。
"""
import numpy as np
from collections import deque

def graph_distance_2d(L, i, j):
    """LxL 周期网格上的最短路径距离（曼哈顿）。"""
    xi, yi = i // L, i % L
    xj, yj = j // L, j % L
    dx = min(abs(xi - xj), L - abs(xi - xj))
    dy = min(abs(yi - yj), L - abs(yi - yj))
    return dx + dy

def euclid_distance_2d(L, i, j):
    """对应连续欧几里得距离。"""
    xi, yi = i // L, i % L
    xj, yj = j // L, j % L
    dx = abs(xi - xj)
    dy = abs(yi - yj)
    return np.sqrt(dx**2 + dy**2)

print("=" * 78)
print("  OP-D4: 图距离 vs 连续测地距离 (2D 周期)")
print("=" * 78)
print()

# 注意: L1 距离和欧几里得距离在 2D 中不同
# 图距离 = L1 距离 (曼哈顿)
# 连续测地距离 = 欧几里得距离

L = 100
print(f"  L = {L}")
print()
print(f"  {'i':>4s}  {'j':>4s}  {'d_graph':>10s}  {'d_euclid':>10s}  {'比值':>10s}")
print("  " + "-" * 50)

origin = 0
for (xi, yi) in [(1,0), (5,0), (10,0), (20,0), (10,10), (20,20), (30,40)]:
    j = xi * L + yi
    d_g = graph_distance_2d(L, origin, j)
    d_e = euclid_distance_2d(L, origin, j)
    ratio = d_g / d_e if d_e > 0 else np.nan
    print(f"  {xi:4d}  {yi:4d}  {d_g:10.4f}  {d_e:10.4f}  {ratio:10.4f}")

print()
print("  注意: L1 距离 (曼哈顿) 与欧几里得距离的比值在 1 到 √2 之间。")
print("  当方向为 (1,0) 时比值 = 1;")
print("  当方向为 (1,1) 时比值 = √2 ≈ 1.414。")
print()

# ── L1 距离的均值 ──
print("─" * 78)
print("  [1] L1 距离的平均值 vs 欧几里得距离")
print("─" * 78)
print()

# 在 L1 球面上, d_graph = r (壳层指标)
# 连续距离 ~ r (在 L1 意义下)
# 但欧几里得距离 ~ r / √2 到 r 之间

for r in [5, 10, 20, 50]:
    # L1 球面上各点的欧几里得距离
    euclid_dists = []
    for x in range(-r, r+1):
        rem = r - abs(x)
        for y in range(-rem, rem+1):
            z_abs = rem - abs(y)
            for z in ([0] if z_abs == 0 else [z_abs, -z_abs]):
                d = np.sqrt(x**2 + y**2 + z**2)
                euclid_dists.append(d)
    euclid_mean = np.mean(euclid_dists)
    ratio = r / euclid_mean
    print(f"  L1 半径 r = {r:3d}: 平均欧几里得距离 = {euclid_mean:.4f}, "
          f"比值 = {ratio:.4f}")

print()
print("  理论: L1 球面的平均欧几里得距离 ~ r / c")
print("  其中 c = 1.5 (近似), 所以比值 ~ 1.5")
print()

# ── 结论 ──
print("=" * 78)
print("  [结论]")
print("=" * 78)
print("""
  1. 图距离 = L1 距离 (曼哈顿距离)
  
  2. 连续测地距离 = 欧几里得距离
  
  3. 两者的比值依赖方向, 范围 [1, √2]。
  
  4. 在 L1 球面上, 平均欧几里得距离 ~ r/1.5。
  
  5. Gromov-Hausdorff 收敛:
     图距离 d_G 与测地距离 d_g 的偏差有界。
     但两者不完全一致 (L1 vs L2)。
     
  6. 关键问题:
     N.E.A. 的图距离是 L1 距离还是欧几里得距离?
     - 如果波前传播是 L1 (曼哈顿), 图距离 = L1
     - 如果波前传播是各向同性, 图距离 ~ 欧几里得
     
     洛伦兹恢复 (OP-D1) 要求各向同性传播。
     所以 N.E.A. 的图距离应该在宏观极限下趋于欧几里得。
""")
print("=" * 78)