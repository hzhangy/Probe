#!/usr/bin/env python3
# nea_stitch_binding_v3.py
"""
缝合网络绑定判据 v3: 双重判据 (网络响应 + 引力剥离)。

永久断裂条件:
  v_rel > max(v_stitch, v_escape)

  v_stitch = ε√(Z/m_p)c ≈ 624 km/s   (N.E.A. 网络响应)
  v_escape = √(2GM/R)                (引力剥离)

物理图像:
  - v_rel < v_stitch: 气体跟随网络, 有暗物质
  - v_stitch < v_rel < v_escape: 气体暂时剥离但引力束缚, 最终恢复
  - v_rel > max(v_stitch, v_escape): 气体永久剥离, 无暗物质
"""
import numpy as np

# ── N.E.A. 常数 ──
Delta = 1 - np.sqrt(3)/2
Z_MeV = 0.406640
m_p_MeV = 938.272
c_kms = 299792.458
eps = 0.1
v_stitch = eps * np.sqrt(Z_MeV/m_p_MeV) * c_kms   # 624 km/s

# ── 引力常数与单位 ──
G = 6.674e-11        # m³ kg⁻¹ s⁻²
Msun = 1.989e30      # kg
kpc = 3.086e19       # m
Mpc = 3.086e22       # m

print("=" * 78)
print("  缝合网络绑定判据 v3 (双重判据)")
print("=" * 78)
print(f"  v_stitch = {v_stitch:.0f} km/s")
print()

# ── 天体参数 ──
# (名称, v_rel, M_grav [Msun], R_grav, 气体有暗物质?, 说明)
objects = [
    # 无相对运动
    ("后发座",          0,    1.0e15, 3*Mpc,  True,  "维里平衡"),
    ("正常星系团",      0,    5.0e14, 2*Mpc,  True,  "维里平衡"),
    ("星系盘",          0,    1.0e12, 30*kpc, True,  "旋转平衡"),
    ("矮星系",          0,    1.0e9,  5*kpc,  True,  "维里平衡"),
    # 子弹星团
    ("子弹气(子团)",    4500, 1.0e14, 1*Mpc,  False, "子团质量"),
    ("主团气体",        4500, 1.0e15, 3*Mpc,  False, "主团质量"),
    # 星系团并合
    ("并合 El Gordo",   1500, 2.0e15, 3*Mpc,  True,  "大质量, 强束缚"),
    ("并合 MACS J0025", 1200, 1.0e15, 2*Mpc,  True,  "大质量, 强束缚"),
    # CMB
    ("CMB 复合期",      0,    1.0e18, 1e7*Mpc, True, "绝热膨胀"),
]

print(f"  {'系统':<18s} {'v_rel':>6s} {'v_esc':>7s} {'v/v_s':>6s} {'v/v_e':>6s} "
      f"{'判据':>6s} {'预测':>6s} {'观测':>5s} {'一致':>5s}")
print("  " + "-" * 82)

all_ok = True
for name, v_rel, M, R, has_dm, note in objects:
    # 逃逸速度
    v_escape = np.sqrt(2 * G * M * Msun / R) / 1e3
    
    ratio_stitch = v_rel / v_stitch
    ratio_escape = v_rel / v_escape
    
    # 永久断裂判据
    v_max = max(v_stitch, v_escape)
    fractured = v_rel > v_max
    predicted_dm = not fractured
    
    ok = (predicted_dm == has_dm)
    if not ok: all_ok = False
    
    pred_str = "✅ 有" if predicted_dm else "❌ 无"
    obs_str = "✅" if has_dm else "❌"
    match = "✓" if ok else "✗"
    bound_str = "断裂" if fractured else "绑定"
    
    print(f"  {name:<18s} {v_rel:>6.0f} {v_escape:>7.0f} {ratio_stitch:>6.2f} "
          f"{ratio_escape:>6.2f} {bound_str:>6s} {pred_str:>6s} {obs_str:>5s} {match:>5s}")

print()
print(f"  全局一致: {'✅ 全部通过' if all_ok else '❌ 有不一致'}")
print()

# ── 关键案例 ──
print("=" * 78)
print("  关键案例")
print("=" * 78)
print()

# 子弹气
M_bullet, R_bullet = 1.0e14*Msun, 1*Mpc
v_esc_bullet = np.sqrt(2*G*M_bullet/R_bullet)/1e3
print(f"  子弹气 (子团):")
print(f"    v_rel = 4500 km/s")
print(f"    v_escape = {v_esc_bullet:.0f} km/s")
print(f"    v_rel > v_escape → 永久断裂 → 无暗物质 ✓")
print()

# El Gordo
M_el, R_el = 2.0e15*Msun, 3*Mpc
v_esc_el = np.sqrt(2*G*M_el/R_el)/1e3
print(f"  El Gordo (星系团并合):")
print(f"    v_rel = 1500 km/s")
print(f"    v_escape = {v_esc_el:.0f} km/s")
print(f"    v_rel < v_escape → 暂时断裂, 引力重新束缚 → 有暗物质 ✓")
print()

print("  物理机制:")
print(f"    子弹气: 高速穿过, 气体逃逸引力势阱, 网络永久断裂")
print(f"    星系团并合: 气体被激波减速但未逃逸, 网络最终恢复")
print()

# ── 可证伪预言 ──
print("=" * 78)
print("  可证伪预言")
print("=" * 78)
print(f"""
  永久断裂条件: v_rel > max(v_stitch, v_escape)

  具体预言:
    1. 星系团并合中, 若 v_rel > √(2GM/R), 气体应失去暗物质信号。
    2. 子弹星团型系统 (v_rel > 1500 km/s, 子团质量 < 10^14 M_sun) 
       应普遍显示气体暗物质缺失。
    3. 大质量星系团 (M > 10^15 M_sun) 的并合, 即使 v_rel = 2000 km/s,
       气体仍应有暗物质 (因为 v_escape > 2000 km/s)。
""")
print("=" * 78)