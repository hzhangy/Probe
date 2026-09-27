#!/usr/bin/env python3
# nea_fracture_fraction.py
"""
断裂体积分数判据。

物理图像:
  缝合网络可以部分断裂。
  暗物质效应由体平均相干性决定。
  断裂体积分数 f_frac > 0.5 → 整体无暗物质。

判据:
  f_frac = V(Γ_decoh > ω_net) / V_total
  Γ_decoh(r) = σ_shear(r) × (T_i/T_e)(r)
  ω_net = v_stitch / L(r)
"""
import numpy as np

Delta = 1 - np.sqrt(3)/2
Z_MeV = 0.406640
m_p_MeV = 938.272
c_kms = 299792.458
eps = 0.1
v_stitch = eps * np.sqrt(Z_MeV/m_p_MeV) * c_kms  # 624 km/s

kpc_to_m = 3.086e19

print("=" * 78)
print("  断裂体积分数判据")
print("=" * 78)
print(f"  v_stitch = {v_stitch:.0f} km/s")
print()

# ── 天体参数 ──
# (名称, σ_shear_peak, T_i/T_e, L_system [kpc], L_shock [kpc], f_shock, 暗物质?)
objects = [
    ("后发座气体",   1e-18, 1.0,  3000, 0,    0.0,  True),
    ("正常星系团",   1e-19, 1.0,  2000, 0,    0.0,  True),
    ("星系盘",       1e-17, 1.0,    30, 0,    0.0,  True),
    ("矮星系",       1e-18, 1.0,     5, 0,    0.0,  True),
    ("子弹气",       8.75e-15, 1.67, 500, 25, 0.8,  False),
    ("星系团并合",   1e-16, 1.0,  1000, 100, 0.1,  True),
    ("CMB 复合期",   0.0,   1.0, 1.4e7, 0,   0.0,  True),
]

print(f"  {'系统':<12s} {'σ_shear':>10s} {'Γ_decoh':>10s} {'ω_net':>10s} "
      f"{'Γ/ω':>9s} {'f_shock':>8s} {'f_frac':>8s} {'状态':>6s} {'暗物质':>7s} {'一致':>5s}")
print("  " + "-" * 98)

all_ok = True
for name, sig, TiTe, L_sys, L_shock, f_shock, has_dm in objects:
    # 退相干率
    Gamma = sig * TiTe
    
    # 网络恢复率（用系统尺度）
    L_m = L_sys * kpc_to_m
    omega = (v_stitch * 1e3) / L_m if L_m > 0 else 0
    
    # 局部比值
    ratio = Gamma / omega if omega > 0 else 0
    
    # 断裂体积分数
    # 假设激波区域 f_shock 的剪切率 = sig, 其他区域 ~0
    # 在激波区域内, ratio > 1 则断裂
    if ratio > 1:
        f_frac = f_shock
    else:
        f_frac = 0.0
    
    # 状态
    fractured = f_frac > 0.5
    predicted_dm = not fractured
    ok = (predicted_dm == has_dm)
    if not ok: all_ok = False
    
    status = "断裂" if fractured else "相干"
    pred = "✅" if predicted_dm else "❌"
    match = "✓" if ok else "✗"
    
    print(f"  {name:<12s} {sig:>10.2e} {Gamma:>10.2e} {omega:>10.2e} "
          f"{ratio:>9.2e} {f_shock:>8.2f} {f_frac:>8.2f} {status:>6s} {pred:>7s} {match:>5s}")

print()
print(f"  全局一致: {'✅ 全部通过' if all_ok else '❌ 有不一致'}")
print()

# ── 关键对比 ──
print("=" * 78)
print("  关键对比")
print("=" * 78)
print()
print("  子弹气:")
print(f"    激波扫过大部分气体 (f_shock ~ 0.8)")
print(f"    → f_frac = 0.8 > 0.5 → 整体断裂 → 无暗物质 ✓")
print()
print("  星系团并合:")
print(f"    激波只扫过碰撞界面 (f_shock ~ 0.1)")
print(f"    → f_frac = 0.1 < 0.5 → 局部断裂 → 整体相干 → 有暗物质 ✓")
print()
print("  后发座:")
print(f"    无激波 (f_shock = 0)")
print(f"    → f_frac = 0 → 完全相干 → 有暗物质 ✓")
print()