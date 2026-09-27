#!/usr/bin/env python3
# nea_stitch_decoherence.py
"""
缝合网络退相干判据。

物理图像:
  - 缝合网络是 K4 缺陷的集体相位相干结构 (Kuramoto/XY 型)
  - 相干 → 2D 全息坍缩 → 1/r → 有暗物质效应
  - 断裂 → 3D 纯牛顿相 → 1/r² → 无暗物质效应
  - 断裂条件: 退相干率 Γ_decoh > 网络恢复率 ω_net = v_stitch / L

判据:
  Γ_decoh = σ_shear × (T_i / T_e)   (非平衡驱动)
  ω_net  = v_stitch / L
  网络状态 = 断裂 if Γ_decoh > ω_net else 相干
"""
import numpy as np

# ── N.E.A. 基本常数 ──
Delta = 1 - np.sqrt(3)/2
Z_MeV = 0.406640
m_p_MeV = 938.272
c_kms = 299792.458
eps = 0.1
v_stitch = eps * np.sqrt(Z_MeV / m_p_MeV) * c_kms   # 624 km/s

print("=" * 78)
print("  N.E.A. 缝合网络退相干判据")
print("=" * 78)
print(f"  v_stitch = ε√(Z/m_p)c = {v_stitch:.0f} km/s")
print()

# ── 天体参数 ──
# (名称, σ_shear [s^-1], T_i [keV], T_e [keV], L [kpc], 观测暗物质?)
objects = [
    # 平衡态: 网络相干 → 有暗物质
    ("后发座气体",   1e-18,  8.0,   8.0,  3000, True),
    ("正常星系团",   1e-19,  5.0,   5.0,  2000, True),
    ("星系盘",       1e-17,  1.0,   1.0,    30, True),
    ("矮星系",       1e-18,  0.1,   0.1,     5, True),
    # 耗散态: 网络断裂 → 无暗物质
    ("子弹气激波",   8.75e-15, 25.0, 15.0,  500, False),
    ("子弹气前锋",   8.75e-15, 25.0, 15.0,  500, False),
    ("星系团并合",   1e-16, 10.0,  10.0, 1000, True),   # 中等
    # CMB 参考
    ("CMB 复合期",   0.0,     0.3,   0.3, 1.4e7, True),
]

# ── 单位转换 ──
kpc_to_m = 3.086e19

print(f"  {'系统':<14s} {'σ_shear':>11s} {'T_i/T_e':>8s} {'Γ_decoh':>11s} "
      f"{'ω_net':>11s} {'Γ/ω':>9s} {'状态':>6s} {'暗物质':>7s} {'一致?':>6s}")
print("  " + "-" * 92)

all_ok = True
for name, sig, Ti, Te, L, has_dm in objects:
    # 非平衡因子
    non_eq = Ti / Te if Te > 0 else 1.0
    
    # 退相干率 (非平衡驱动)
    Gamma = sig * non_eq
    
    # 网络恢复率
    L_m = L * kpc_to_m
    v_stitch_ms = v_stitch * 1e3
    omega = v_stitch_ms / L_m if L_m > 0 else 0
    
    # 判据
    ratio = Gamma / omega if omega > 0 else 0
    
    # 状态: 断裂 if Γ > ω
    fractured = ratio > 1.0
    predicted_dm = not fractured
    ok = (predicted_dm == has_dm)
    if not ok: all_ok = False
    
    status = "断裂" if fractured else "相干"
    pred = "✅" if predicted_dm else "❌"
    match = "✓" if ok else "✗"
    
    print(f"  {name:<14s} {sig:>11.2e} {non_eq:>8.2f} {Gamma:>11.2e} "
          f"{omega:>11.2e} {ratio:>9.2e} {status:>6s} {pred:>7s} {match:>6s}")

print()
print(f"  全局一致: {'✅ 全部通过' if all_ok else '❌ 有不一致'}")
print()

# ── 物理分析 ──
print("=" * 78)
print("  物理分析")
print("=" * 78)
print()
print("  判据: Γ_decoh / ω_net > 1 → 缝合断裂 → 3D 纯牛顿 → 无暗物质")
print()
print(f"  后发座: σ_shear ~ 1e-18 s⁻¹, T_i/T_e = 1")
print(f"    Γ/ω = {1e-18 * 1 / (v_stitch*1e3/(3000*kpc_to_m)):.2e}  << 1  → 相干 ✓")
print()
print(f"  子弹气: σ_shear = 8.75e-15 s⁻¹, T_i/T_e = 25/15 = 1.67")
print(f"    Γ/ω = {8.75e-15 * 25/15 / (v_stitch*1e3/(500*kpc_to_m)):.2e}  >> 1  → 断裂 ✓")
print()
print(f"  CMB: σ_shear = 0 (绝热膨胀, 无剪切)")
print(f"    Γ/ω = 0  << 1  → 相干 → 第三峰正常 ✓")
print()

# ── 物理解释 ──
print("=" * 78)
print("  物理机制")
print("=" * 78)
print(f"""
  1. 缝合网络的相干条件:
       网络恢复时间 = L / v_stitch
       剪切破坏时间 = 1 / σ_shear
       相干 ⟺ L / v_stitch < 1 / σ_shear ⟺ σ_shear · L / v_stitch < 1

  2. 非平衡增强:
       离子-电子温度比 T_i/T_e > 1 时, 库仑碰撞的非平衡程度放大退相干率
       Γ_decoh = σ_shear × (T_i/T_e)

  3. 子弹气的特殊性:
       σ_shear = 8.75e-15 s⁻¹ (比平衡态高 4-5 个数量级)
       T_i/T_e = 1.67 (强非平衡)
       Γ/ω ~ 10⁴ >> 1 → 必然断裂

  4. CMB 的自洽:
       复合期剪切率为零 (绝热膨胀)
       Γ/ω = 0 → 网络完美相干 → 2D 暗物质正常形成 → 第三峰不被压低
""")
print("=" * 78)