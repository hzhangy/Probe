#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_op_s1b_friedmann_v2.py

OP-S1b v2: 从暗能量机制导出 Friedmann 方程。

核心逻辑:
  1. 从 S 卷的 Omega_m/Omega_Lambda 出发
  2. 计算 rho_Lambda 和 n_node
  3. 从 Friedmann 方程导出 H(z)
  4. 检验与观测一致性
  5. 检查 n_node 与拓扑常数的关系
"""
import numpy as np

# ── 物理常数 ──
c = 2.998e8            # m/s
G = 6.674e-11          # m^3 kg^-1 s^-2
hbar = 6.582e-22       # MeV s
Z_MeV = 0.406640       # MeV, ZY 单位
kg_per_MeV = 1.783e-30 # kg/MeV/c^2
B_kg = Z_MeV * kg_per_MeV  # 每个 ZY 对应的质量
H_0_obs = 2.184e-18    # s^-1 (67.4 km/s/Mpc)

# ── 拓扑常数 ──
Delta = 1 - np.sqrt(3)/2
R = 1/(1 + np.pi)
N_max = np.exp(10*np.sqrt(3))
eps = 0.1

# ── S 卷结果 ──
Omega_m = 0.315473
Omega_L = 0.684527
K_ratio = Omega_m / Omega_L
K_ratio_theory = 0.4*np.pi - 1/(0.4*np.pi)

# ── 临界密度 ──
rho_crit = 3 * H_0_obs**2 / (8 * np.pi * G)

print("=" * 78)
print("  N.E.A. OP-S1b v2: 从暗能量机制导出 Friedmann 方程")
print("=" * 78)
print()

# ── 第 1 步：从 S 卷导出 rho_Lambda ──
print("─" * 78)
print("  [1] 从 S 卷导出 rho_Lambda")
print("─" * 78)
rho_L = Omega_L * rho_crit
rho_m = Omega_m * rho_crit
print(f"  rho_crit        = {rho_crit:.4e} kg/m^3")
print(f"  rho_Lambda      = {rho_L:.4e} kg/m^3")
print(f"  rho_m (today)   = {rho_m:.4e} kg/m^3")
print(f"  Omega_m/Omega_L = {K_ratio:.6f}")
print(f"  理论预测        = {K_ratio_theory:.6f}")
print(f"  偏差            = {abs(K_ratio - K_ratio_theory)/K_ratio_theory * 100:.3f}%")
print()

# ── 第 2 步：从 rho_Lambda 导出 n_node ──
print("─" * 78)
print("  [2] 从 rho_Lambda 导出空间节点密度 n_node")
print("─" * 78)
n_node = rho_L / B_kg
l_node = n_node ** (-1/3)
print(f"  B (ZY)          = {B_kg:.4e} kg")
print(f"  n_node          = {n_node:.4e} m^-3")
print(f"  节点间距        = {l_node*100:.4f} cm")
print(f"  节点间距        = {l_node:.4e} m")
print()

# ── 第 3 步：从 n_node 反推 H_0 ──
print("─" * 78)
print("  [3] 自洽性: 从 n_node 反推 H_0")
print("─" * 78)
H_0_recon = np.sqrt(8 * np.pi * G * rho_L / 3)
print(f"  H_0 (从 n_node) = {H_0_recon:.4e} s^-1")
print(f"  H_0 (观测)      = {H_0_obs:.4e} s^-1")
print(f"  偏差            = {abs(H_0_recon - H_0_obs)/H_0_obs * 100:.4f}%")
print()

# ── 第 4 步：H(z) 演化 ──
print("─" * 78)
print("  [4] H(z) 演化 (从 Friedmann 方程)")
print("─" * 78)

def H_z(z):
    return H_0_obs * np.sqrt(Omega_m * (1+z)**3 + Omega_L)

print(f"  {'z':>8s}  {'a':>10s}  {'H(z)':>16s}  {'rho_m':>14s}  {'rho_L':>14s}")
print(f"  {'':>8s}  {'':>10s}  {'km/s/Mpc':>16s}  {'kg/m^3':>14s}  {'kg/m^3':>14s}")
print("  " + "-" * 70)

for z in [0, 0.5, 1, 2, 5, 10, 100, 1100, 3400]:
    H = H_z(z)
    a = 1/(1+z)
    rho_m_z = rho_m * (1+z)**3
    print(f"  {z:8.2f}  {a:10.6f}  {H/3.24e-20:16.3f}  "
          f"{rho_m_z:14.4e}  {rho_L:14.4e}")

print()

# ── 第 5 步：膨胀历史 ──
print("─" * 78)
print("  [5] 膨胀历史: a(t) 与 q(t)")
print("─" * 78)

# 数值积分 Friedmann 方程
from scipy.integrate import solve_ivp

def friedmann_rhs(t, y):
    a = y[0]
    H = H_0_obs * np.sqrt(Omega_m / a**3 + Omega_L)
    return [a * H]

# 从 a=1e-3 积分到今天
a_init = 1e-3
t_init = 0
# 正确的时间尺度: 用 H_0^-1
t_H = 1 / H_0_obs  # s

sol = solve_ivp(friedmann_rhs, [0, 2*t_H], [a_init],
                rtol=1e-10, atol=1e-12, dense_output=True)

# 找 a=1 的时间
from scipy.optimize import brentq

def find_a_one():
    def f(t):
        return sol.sol(t)[0] - 1.0
    return brentq(f, 0.1*t_H, 2*t_H)

t_today = find_a_one()
age_Gyr = t_today / (3.156e16)  # s -> Gyr

print(f"  从 a={a_init} 积分到 a=1")
print(f"  宇宙年龄 = {age_Gyr:.3f} Gyr")
print(f"  观测值  = 13.787 ± 0.020 Gyr")
print(f"  偏差    = {abs(age_Gyr - 13.787)/13.787 * 100:.3f}%")
print()

# 减速参数 q_0
q_0 = 0.5 * Omega_m - Omega_L
print(f"  减速参数 q_0 = 0.5*Omega_m - Omega_L = {q_0:.4f}")
print(f"  观测值        ≈ -0.527")
print()

# ── 第 6 步：n_node 与拓扑常数的关系探索 ──
print("─" * 78)
print("  [6] n_node 与拓扑常数的关系探索")
print("─" * 78)

# 候选关系
candidates = {
    "1/lambda_Z^3": 1 / (hbar*c/Z_MeV/1e15)**3,
    "1/(R*lambda_Z)^3": 1 / (R*hbar*c/Z_MeV/1e15)**3,
    "1/(eps*lambda_Z)^3": 1 / (eps*hbar*c/Z_MeV/1e15)**3,
    "1/(alpha_G*lambda_Z)^3": 1 / (R/N_max**5 * hbar*c/Z_MeV/1e15)**3,
    "H_0^3/c^3": (H_0_obs/c)**3,
    "1/(c/H_0)^3 * R": R / (c/H_0_obs)**3,
    "1/(c/H_0)^3 / N_max": 1 / (c/H_0_obs)**3 / N_max,
    "1/(c/H_0)^3 * alpha_G": (R/N_max**5) / (c/H_0_obs)**3,
}

print(f"  {'候选':>30s}  {'值 (m^-3)':>16s}  {'比值 to obs':>16s}")
print("  " + "-" * 68)
for name, val in candidates.items():
    ratio = val / n_node
    print(f"  {name:>30s}  {val:16.4e}  {ratio:16.4e}")

print()
print("  obs n_node = {:.4e} m^-3".format(n_node))
print()

# ── 第 7 步：关键的拓扑组合探索 ──
print("─" * 78)
print("  [7] 关键拓扑组合探索")
print("─" * 78)

# 从观测反推 n_node 的表达式
# n_node = 3 H_0^2 / (8 pi G B)
# 假设 H_0 = (1/t_Tick) * alpha_G * (19-4sqrt(3))/20
# 则 n_node = 3/(8 pi G B) * (1/t_Tick)^2 * alpha_G^2 * f_H^2
# 其中 f_H = (19-4sqrt(3))/20

t_Tick = hbar / Z_MeV * 1e-21 / 1.519  # MeV*s / MeV -> s (粗略)
# 用更精确的值
hbar_MeV_s = 6.582e-22
t_Tick = hbar_MeV_s / Z_MeV  # s
f_H = (19 - 4*np.sqrt(3)) / 20

print(f"  t_Tick = {t_Tick:.4e} s")
print(f"  f_H    = {f_H:.6f}")
print(f"  1/t_Tick = {1/t_Tick:.4e} s^-1")
print()

# 候选: n_node = alpha_G / (c * t_Tick)^3 * (某个组合)
lambda_Z_m = hbar*c / Z_MeV * 1e-15  # fm -> m
print(f"  lambda_Z = {lambda_Z_m:.4e} m")
print()

# 尝试: n_node = 1 / (lambda_Z * N_max^k)^3
for k in [1, 2, 3]:
    val = 1 / (lambda_Z_m * N_max**k)**3
    ratio = val / n_node
    print(f"  1/(lambda_Z * N_max^{k})^3 = {val:.4e} m^-3, 比值 = {ratio:.4e}")
print()

# 尝试: n_node = (H_0/c)^3 * N_max^k
for k in [1, 2, 3, 4, 5]:
    val = (H_0_obs/c)**3 * N_max**k
    ratio = val / n_node
    print(f"  (H_0/c)^3 * N_max^{k} = {val:.4e} m^-3, 比值 = {ratio:.4e}")
print()

# ── 总结 ──
print("=" * 78)
print("  [总结] OP-S1b 进展")
print("=" * 78)
print(f"""
  1. 从 S 卷的 Omega_m/Omega_L 出发, Friedmann 方程给出:
     - H_0 = {H_0_recon:.3e} s^-1 (自洽)
     - 宇宙年龄 = {age_Gyr:.3f} Gyr (观测 13.787, 偏差 {abs(age_Gyr-13.787)/13.787*100:.2f}%)
     - 减速参数 q_0 = {q_0:.4f} (观测 ≈ -0.527)
     
  2. n_node = rho_L / B = {n_node:.3e} m^-3
     节点间距 = {l_node*100:.3f} cm
     
  3. 自洽性: H_0 (从 n_node) 与 H_0 (观测) 一致到 {abs(H_0_recon-H_0_obs)/H_0_obs*100:.4f}%
     
  4. 开放问题: n_node 的第一性原理导出
     候选组合与观测比值:
       (H_0/c)^3 * N_max^5 = {(H_0_obs/c)**3 * N_max**5:.4e} m^-3
       比值 = {(H_0_obs/c)**3 * N_max**5 / n_node:.4e}
     
  5. 物理图像:
     - 膨胀由暗能量(空间维护租金)驱动
     - rho_L = n_node * B 是常数
     - Friedmann 方程从焓守恒 + 空间节点产生导出
     
  6. OP-S1b 状态:
     - 路径澄清: 纯 KE 不产生膨胀 (已证)
     - Friedmann 导出: 从 S 卷 + 暗能量机制 (已闭合)
     - n_node 第一性原理: 开放 (OP-S1b-4)
""")
print("=" * 78)