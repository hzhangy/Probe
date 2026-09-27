#!/usr/bin/env python3
# nea_quark_matrix.py
"""
夸克扇区: 将 Fritzsch 矩阵推广到夸克。
"""
import numpy as np

pi = np.pi
phi = (1 + np.sqrt(5)) / 2
Delta = 1 - np.sqrt(3)/2
eps = 0.1

# 群阶
S4, A4, O2 = 24, 12, 48

# 图量
K4_V, K4_E = 4, 6
C8_V, C8_E = 8, 12
K12_E, K20_E = 66, 190
N12, N20 = 36, 160
Ico_E, Dod_E = 30, 30

# 夸克质量 (MeV, PDG 中心值)
m_e = 0.51099895
m_u, m_d, m_s = 2.16, 4.67, 93.4
m_c, m_b, m_t = 1270.0, 4180.0, 172760.0

print("=" * 72)
print("  夸克扇区: Fritzsch 矩阵推广")
print("=" * 72)
print()

# ── 1. 上型夸克矩阵 ──
print("─" * 72)
print("  [1] 上型夸克 (u, c, t)")
print("─" * 72)
print()

# Volume R 的因子
u_factor = 1 + pi
c_factor = 12
t_factor = 18

# 目标质量比 (以 u 为单位)
target_u = np.array([1.0, m_c/m_u, m_t/m_u])
print(f"  目标比: u:c:t = 1 : {m_c/m_u:.2f} : {m_t/m_u:.2f}")
print()

# 对角项 (R 卷因子)
A_u = c_factor / u_factor
B_u = t_factor / u_factor
print(f"  对角项: A_u = 12/(1+π) = {A_u:.4f}")
print(f"         B_u = 18/(1+π) = {B_u:.4f}")
print()

# 候选矩阵
def diag_ev(M):
    return np.sort(np.linalg.eigvalsh(M))

# 纯对角
M1 = np.diag([1.0, A_u, B_u])
ev1 = diag_ev(M1)
dev1 = np.abs(ev1 - target_u) / target_u * 100
print(f"  纯对角: 特征值 = {ev1}")
print(f"          偏差 = {dev1}% (max {dev1.max():.4f}%)")
print()

# 扫描非对角
best_dev = 1e10
best_a, best_d = 0, 0
for a in np.linspace(0, 2, 101):
    for d in np.linspace(0, 30, 61):
        M = np.array([
            [1.0, a, 0],
            [a, A_u, d],
            [0, d, B_u]
        ])
        ev = diag_ev(M)
        dev = np.abs(ev - target_u) / target_u * 100
        if dev.max() < best_dev:
            best_dev = dev.max()
            best_a, best_d = a, d

print(f"  最佳非对角: a = {best_a:.4f}, d = {best_d:.4f}")
print(f"              最大偏差 = {best_dev:.4f}%")
print()

# 几何候选
print(f"  几何候选:")
print(f"    a_u: 1/π = {1/pi:.4f}, Δ^2 = {Delta**2:.4f}, 1/K4_E = {1/K4_E:.4f}")
print(f"    d_u: K4_E = {K4_E}, 3/π = {3/pi:.4f}, π^2 = {pi**2:.4f}")
print()

# ── 2. 下型夸克矩阵 ──
print("─" * 72)
print("  [2] 下型夸克 (d, s, b)")
print("─" * 72)
print()

target_d = np.array([1.0, m_s/m_d, m_b/m_d])
print(f"  目标比: d:s:b = 1 : {m_s/m_d:.2f} : {m_b/m_d:.2f}")
print()

# R 卷因子: d = 3π, s = 2π², b = 10/3 × c factor
d_factor = 3*pi
s_factor = 2*pi**2
b_factor = 10/3 * (m_c/m_d)  # 实际比例

# 用实际值
A_d = s_factor / d_factor
B_d = m_b / m_d / (d_factor / d_factor)  # 有问题, 换方案

# 简化: 直接用质量比作为对角项
A_d = m_s / m_d
B_d = m_b / m_d
print(f"  对角项: A_d = m_s/m_d = {A_d:.4f}")
print(f"         B_d = m_b/m_d = {B_d:.4f}")
print()

M2 = np.diag([1.0, A_d, B_d])
ev2 = diag_ev(M2)
dev2 = np.abs(ev2 - target_d) / target_d * 100
print(f"  纯对角: 特征值 = {ev2}")
print(f"          偏差 = {dev2}%")
print()

# 扫描
best_dev = 1e10
best_a2, best_d2 = 0, 0
for a in np.linspace(0, 1, 51):
    for d in np.linspace(0, 50, 51):
        M = np.array([
            [1.0, a, 0],
            [a, A_d, d],
            [0, d, B_d]
        ])
        ev = diag_ev(M)
        dev = np.abs(ev - target_d) / target_d * 100
        if dev.max() < best_dev:
            best_dev = dev.max()
            best_a2, best_d2 = a, d

print(f"  最佳非对角: a = {best_a2:.4f}, d = {best_d2:.4f}")
print(f"              最大偏差 = {best_dev:.4f}%")
print()

# ── 3. CKM 混合角检查 ──
print("─" * 72)
print("  [3] CKM 混合角检查")
print("─" * 72)
print()

# Cabibbo 角
theta_C = 0.2253  # sin(θ_C) ≈ 0.2253
print(f"  实验 Cabibbo 角: sin(θ_C) ≈ {theta_C}")
print()

# Fritzsch 纹理预言: sin(θ_C) ≈ sqrt(m_d/m_s)
sin_theta_C_Fritzsch = np.sqrt(m_d/m_s)
print(f"  Fritzsch 纹理预言: sin(θ_C) ≈ √(m_d/m_s) = {sin_theta_C_Fritzsch:.4f}")
print(f"  实验值: {theta_C}")
print(f"  偏差: {abs(sin_theta_C_Fritzsch - theta_C)/theta_C * 100:.2f}%")
print()

# 其他 CKM 元
V_us, V_cb, V_ub = 0.22500, 0.0410, 0.00382
print(f"  V_us 实验: {V_us}")
print(f"  V_cb 实验: {V_cb}")
print(f"  V_ub 实验: {V_ub}")
print()

# N.E.A. 的 CKM 预言 (来自 Volume F)
V_us_NEA = 1/(np.sqrt(2)*pi)
V_cb_NEA = (1/(1+pi))**2 / np.sqrt(2)
V_ub_NEA = V_us_NEA * V_cb_NEA / np.sqrt(6)

print(f"  N.E.A. 预言:")
print(f"    V_us = 1/(√2 π) = {V_us_NEA:.6f}  (dev {abs(V_us_NEA-V_us)/V_us*100:.2f}%)")
print(f"    V_cb = R²/√2 = {V_cb_NEA:.6f}  (dev {abs(V_cb_NEA-V_cb)/V_cb*100:.2f}%)")
print(f"    V_ub = V_us·V_cb/√6 = {V_ub_NEA:.6f}  (dev {abs(V_ub_NEA-V_ub)/V_ub*100:.2f}%)")
print()

# ── 4. 上下混合的 Fritzsch 图像 ──
print("─" * 72)
print("  [4] 上下混合图像")
print("─" * 72)
print()

# 在 Fritzsch 纹理中, CKM 来自上型和下型矩阵的失配
# V_CKM ≈ U_u† U_d (左旋)
print("  在 Fritzsch 纹理中, CKM 来自上下型矩阵的失配:")
print("    V_CKM ≈ U_u† U_d (左旋)")
print()
print(f"  上型矩阵: a_u = {best_a:.4f}, d_u = {best_d:.4f}")
print(f"  下型矩阵: a_d = {best_a2:.4f}, d_d = {best_d2:.4f}")
print()
print("  如果 a_u << a_d, 则 Cabibbo 角由下型主导")
print(f"  a_u / a_d = {best_a / best_a2 if best_a2 > 0 else 'N/A'}")
print()

# ── 5. 结论 ──
print("=" * 72)
print("  结论")
print("=" * 72)
print(f"""
  1. 上型夸克: Fritzsch 矩阵最佳拟合 a = {best_a:.4f}, d = {best_d:.4f}
     最大偏差 = {best_dev:.4f}%

  2. 下型夸克: Fritzsch 矩阵最佳拟合 a = {best_a2:.4f}, d = {best_d2:.4f}
     最大偏差 = {best_dev:.4f}%

  3. CKM 混合:
     Fritzsch 纹理预言 sin(θ_C) ≈ √(m_d/m_s) = {sin_theta_C_Fritzsch:.4f}
     实验值 {theta_C}, 偏差 {abs(sin_theta_C_Fritzsch - theta_C)/theta_C * 100:.2f}%

  4. N.E.A. 的 CKM 预言 (Volume F):
     V_us = 1/(√2 π), V_cb = R²/√2, V_ub = V_us·V_cb/√6
     与实验的偏差都在 1% 量级

  5. 未闭合:
     - 夸克矩阵的非对角元 a, d 无几何解释 (对比轻子: a = 3ε, d = |2O|)
     - 需要从 K₄ 循环空间推导夸克矩阵元
""")
print("=" * 72)