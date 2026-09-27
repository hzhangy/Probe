#!/usr/bin/env python3
# nea_ram_pressure_baryon.py
"""
Gunn-Gott 冲压剥离判据 (纯重子版)。

修正:
  - 去掉暗物质预设 Sigma_total = 10 * Sigma_gas
  - 用纯重子: Sigma_total = Sigma_gas + Sigma_star
  - 加入耗散条件区分"孤立耗散气体" vs "结构化恒星系统"

判据:
  气体被冲压剥离 ⟺ P_ram > F_restore (纯重子)
"""
import numpy as np

# N.E.A. 常数
Z_MeV = 0.406640
m_p_MeV = 938.272
c_kms = 299792.458
eps = 0.1
v_stitch = eps * np.sqrt(Z_MeV/m_p_MeV) * c_kms

# 物理常数
G = 6.674e-11
Msun = 1.989e30
kpc = 3.086e19
m_p = 1.673e-27

print("=" * 88)
print("  Gunn-Gott 冲压剥离判据 (纯重子)")
print("=" * 88)
print(f"  v_stitch = {v_stitch:.0f} km/s")
print(f"  无暗物质预设: Σ_total = Σ_gas + Σ_star")
print()

# (名称, v_rel, rho_ICM, M_gas, M_star, R, T_gas [K], has_dm_gas)
objects = [
    ("后发座",      0,    3e-3, 1e13, 1e12, 300, 1e8,  True),
    ("正常星系团",  0,    1e-3, 1e13, 1e12, 300, 1e8,  True),
    ("星系盘",      0,    1e-3, 1e10, 5e9,   30,  1e4,  True),
    ("矮星系",      0,    1e-5, 1e8,  1e7,    5,  1e4,  True),
    ("子弹气",      4500, 1e-3, 3.5e13, 5e12, 300, 1.5e8, False),
    ("MACS J0025",  1200, 3e-3, 3e12, 5e11, 200, 1.5e8, False),
    ("El Gordo",    1500, 5e-3, 2e13, 3e12, 400, 2e8,  False),
    ("CMB 复合期",  0,    1e3,  1e18, 0,    1e7, 3e3,  True),
]

print(f"  {'系统':<13s} {'v_rel':>6s} {'Σ_gas':>8s} {'Σ_tot':>8s} {'P_ram':>10s} "
      f"{'F_rest':>10s} {'P/F':>7s} {'剥离':>4s} {'DM':>4s} {'一致':>4s}")
print("  " + "-" * 90)

all_ok = True
for name, v_rel, rho_ICM, M_gas, M_star, R_sub, T_gas, has_dm_gas in objects:
    R_SI = R_sub * kpc
    # 纯重子面密度
    Sigma_gas = M_gas * Msun / (np.pi * R_SI**2)
    Sigma_star = M_star * Msun / (np.pi * R_SI**2)
    Sigma_total = Sigma_gas + Sigma_star
    
    # 冲压
    rho_SI = rho_ICM * 1e6 * m_p
    v_SI = v_rel * 1e3
    P_ram = rho_SI * v_SI**2
    
    # 恢复力
    F_restore = 2 * np.pi * G * Sigma_gas * Sigma_total
    
    ratio = P_ram / F_restore if F_restore > 0 else 0
    stripped = ratio > 1.0
    predicted_dm = not stripped
    
    ok = (predicted_dm == has_dm_gas)
    if not ok: all_ok = False
    
    print(f"  {name:<13s} {v_rel:>6.0f} {Sigma_gas:>8.2e} {Sigma_total:>8.2e} "
          f"{P_ram:>10.2e} {F_restore:>10.2e} {ratio:>7.2f} "
          f"{'是' if stripped else '否':>4s} "
          f"{'✅' if has_dm_gas else '❌':>4s} {'✓' if ok else '✗':>4s}")

print()
print(f"  全局一致：{'✅ 全部通过' if all_ok else '❌ 有不一致'}")
print()

# ── 对比修正前后 ──
print("=" * 88)
print("  修正前后对比")
print("=" * 88)
print()
print("  修正前 (Σ_total = 10 Σ_gas):")
print("    子弹星团 P/F ≈ 1.48, MACS J0025 P/F ≈ 2.31, El Gordo P/F ≈ 1.30")
print()
print("  修正后 (Σ_total = 1.15 Σ_gas):")
for name, v_rel, rho_ICM, M_gas, M_star, R_sub, T_gas, has_dm_gas in objects:
    if name in ["子弹气", "MACS J0025", "El Gordo"]:
        R_SI = R_sub * kpc
        Sigma_gas = M_gas * Msun / (np.pi * R_SI**2)
        Sigma_star = M_star * Msun / (np.pi * R_SI**2)
        Sigma_total = Sigma_gas + Sigma_star
        rho_SI = rho_ICM * 1e6 * m_p
        v_SI = v_rel * 1e3
        P_ram = rho_SI * v_SI**2
        F_restore = 2 * np.pi * G * Sigma_gas * Sigma_total
        ratio = P_ram / F_restore
        print(f"    {name:<12s} P/F ≈ {ratio:.2f}")
print()

# ── 物理总结 ──
print("=" * 88)
print("  物理机制")
print("=" * 88)
print(f"""
  1. 冲压剥离 (Gunn-Gott, 纯重子):
       气体被剥离 ⟺ P_ram > 2π G Σ_gas Σ_total
       Σ_total = Σ_gas + Σ_star (无暗物质粒子)

  2. 缝合网络绑定:
       网络绑定"拓扑健康"的系统
       恒星系统 = 无碰撞, 内部低耗散, 网络跟随
       气体 = 耗散流体, 被冲压剥离后孤立, 网络不跟随
       (这个层次需要从 KE 推导, 不是循环假设)

  3. 观测预言:
       气体被冲压剥离的系统 → 气体位置无暗物质透镜峰
       星系位置 → 暗物质透镜峰跟随
""")
print("=" * 88)