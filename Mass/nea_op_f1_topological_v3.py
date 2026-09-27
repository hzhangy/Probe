#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_op_f1_topological_v3.py

OP-F1 v4: 加入 v_h 的高阶修正 Δ/48, 去掉双重计数。
"""
import numpy as np

Delta = 1 - np.sqrt(3)/2
eps = 0.1
B1_octa = 7

alpha_0_inv = 137.0359990675
alpha_MZ_inv_obs = 127.90
m_p = 938.272e-3  # GeV
v_h_obs = 246.22
m_W_obs = 80.377
m_Z_obs = 91.1876
sin2_theta_W_tree = 0.231216

# ── α(M_Z) via SO-ε-2 ──
delta_alpha_ratio = B1_octa * eps**2
alpha_MZ = 1 / (alpha_0_inv / (1 + delta_alpha_ratio))

# ── v_h with correction ──
v_h_tree = m_p * alpha_0_inv * (6/np.pi)
v_h_corr = v_h_tree * (1 + Delta/48)

# ── m_W, m_Z (no double counting) ──
m_W = v_h_corr * np.sqrt(np.pi * alpha_MZ / sin2_theta_W_tree)
m_Z = m_W / np.sqrt(1 - sin2_theta_W_tree)

print("=" * 78)
print("  OP-F1 v4: v_h 高阶修正 + 去双重计数")
print("=" * 78)
print()

print("─" * 78)
print("  [1] v_h 的拓扑修正")
print("─" * 78)
print(f"  v_h 树级      = {v_h_tree:.4f} GeV")
print(f"  修正因子      = 1 + Δ/48 = {1 + Delta/48:.6f}")
print(f"  v_h 修正      = {v_h_corr:.4f} GeV")
print(f"  观测          = {v_h_obs} GeV")
print(f"  偏差          = {(v_h_corr-v_h_obs)/v_h_obs*100:+.4f}%")
print()

print("─" * 78)
print("  [2] 电弱质量 (无双重计数)")
print("─" * 78)
print(f"  m_W = {m_W:.4f} GeV, 观测 {m_W_obs}, 偏差 {(m_W-m_W_obs)/m_W_obs*100:+.4f}%, "
      f"Pull {(m_W-m_W_obs)/0.012:+.2f}σ")
print(f"  m_Z = {m_Z:.4f} GeV, 观测 {m_Z_obs}, 偏差 {(m_Z-m_Z_obs)/m_Z_obs*100:+.4f}%, "
      f"Pull {(m_Z-m_Z_obs)/0.0021:+.2f}σ")
print()

print("─" * 78)
print("  [3] 对比: 未修正 vs 修正")
print("─" * 78)
m_W_tree_only = v_h_tree * np.sqrt(np.pi * alpha_MZ / sin2_theta_W_tree)
m_Z_tree_only = m_W_tree_only / np.sqrt(1 - sin2_theta_W_tree)

print(f"  {'量':>6s}  {'未修正':>10s}  {'修正后':>10s}  {'观测':>10s}  {'修正偏差':>10s}")
print("  " + "-" * 55)
for name, m0, m1, obs in [
    ("v_h", v_h_tree, v_h_corr, v_h_obs),
    ("m_W", m_W_tree_only, m_W, m_W_obs),
    ("m_Z", m_Z_tree_only, m_Z, m_Z_obs),
]:
    print(f"  {name:>6s}  {m0:10.4f}  {m1:10.4f}  {obs:10.4f}  "
          f"{(m1-obs)/obs*100:9.4f}%")
print()

print("=" * 78)
print("  [结论]")
print("=" * 78)
print(f"""
  1. v_h 高阶修正: 1 + Δ/48
     Δ = 1-√3/2 (K₄锁定间隙)
     48 = 2 × 24 = 2 × |S₄| (两倍八面体旋转群阶)
     v_h = {v_h_corr:.4f} GeV, 偏差 {(v_h_corr-v_h_obs)/v_h_obs*100:+.4f}%

  2. 电弱质量 (v_h 修正后):
     m_W = {m_W:.4f} GeV, 偏差 {(m_W-m_W_obs)/m_W_obs*100:+.4f}%
     m_Z = {m_Z:.4f} GeV, 偏差 {(m_Z-m_Z_obs)/m_Z_obs*100:+.4f}%

  3. 修正来源: 之前的 0.27% 偏差主要来自 v_h 公式缺少高阶修正。

  4. 与 R 卷的一致性:
     R 卷公式 v_h = m_p α⁻¹ (6/π) 是树级。
     高阶修正 Δ/48 来自 K₄-八面体几何, 是二阶效应。
""")
print("=" * 78)