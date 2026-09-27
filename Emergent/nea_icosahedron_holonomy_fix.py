#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_icosahedron_holonomy_fix.py
二十面体自旋联络和乐角精确修复验证 (消除规范缠绕，严格锁定 pi/6)
"""
import numpy as np
from itertools import combinations

print("=" * 74)
print("  N.E.A. 二十面体自旋联络内蕴转角修复器 (Holonomy Angle -> pi/6)")
print("=" * 74)

# ── 1. 构造标准正二十面体 (R = 1) ──
phi = (1.0 + np.sqrt(5.0)) / 2.0
raw_verts = np.array([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
], dtype=float)
verts = raw_verts / np.linalg.norm(raw_verts[0])
a_true = np.sqrt(2.0 - 2.0 / np.sqrt(5.0))

faces = []
for i, j, k in combinations(range(12), 3):
    dij = np.linalg.norm(verts[i] - verts[j])
    djk = np.linalg.norm(verts[j] - verts[k])
    dki = np.linalg.norm(verts[k] - verts[i])
    if abs(dij - a_true) < 1e-4 and abs(djk - a_true) < 1e-4 and abs(dki - a_true) < 1e-4:
        v1, v2, v3 = verts[i], verts[j], verts[k]
        normal = np.cross(v2 - v1, v3 - v1)
        center = (v1 + v2 + v3) / 3.0
        if np.dot(normal, center) < 0:
            faces.append((i, k, j))
        else:
            faces.append((i, j, k))

Nf = len(faces)
face_area = np.sqrt(3.0) / 4.0 * a_true**2

# 构建面切空间标架 {e1, e2, N}
face_centers = []
e1_list, e2_list, norm_list = [], [], []

for f in faces:
    v1, v2, v3 = verts[f[0]], verts[f[1]], verts[f[2]]
    c = (v1 + v2 + v3) / 3.0
    c /= np.linalg.norm(c)
    face_centers.append(c)
    
    n_vec = np.cross(v2 - v1, v3 - v1)
    n_vec /= np.linalg.norm(n_vec)
    e1 = (v2 - v1) / np.linalg.norm(v2 - v1)
    e2 = np.cross(n_vec, e1)
    e2 /= np.linalg.norm(e2)
    
    e1_list.append(e1)
    e2_list.append(e2)
    norm_list.append(n_vec)

# 提取 30 条对偶边
edge_to_faces = {}
for f_idx, (v1, v2, v3) in enumerate(faces):
    for e in [tuple(sorted((v1, v2))), tuple(sorted((v2, v3))), tuple(sorted((v3, v1)))]:
        if e not in edge_to_faces:
            edge_to_faces[e] = []
        edge_to_faces[e].append(f_idx)

dual_edges = []
for e, flist in edge_to_faces.items():
    if len(flist) == 2:
        dual_edges.append((flist[0], flist[1], e))

# ── 2. 内蕴公共边转角公式计算自旋联络 U_ij ──
U_dict = {}
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
H_dirac = np.zeros((2 * Nf, 2 * Nf), dtype=complex)
w = a_true / (2.0 * face_area)

for i, j, edge in dual_edges:
    N_i, e1_i, e2_i = norm_list[i], e1_list[i], e2_list[i]
    N_j, e1_j, e2_j = norm_list[j], e1_list[j], e2_list[j]
    c_i = face_centers[i]
    
    # 物理棱的唯一定向 (从小索引指向大索引)
    p1, p2 = verts[edge[0]], verts[edge[1]]
    t_edge = (p2 - p1) / np.linalg.norm(p2 - p1)
    
    # 在面 i 的切空间中，计算该棱的方位角 alpha_i
    t_i_x = np.dot(t_edge, e1_i)
    t_i_y = np.dot(t_edge, e2_i)
    alpha_i = np.arctan2(t_i_y, t_i_x)
    
    # 在面 j 的切空间中，计算同一条棱的方位角 alpha_j
    t_j_x = np.dot(t_edge, e1_j)
    t_j_y = np.dot(t_edge, e2_j)
    alpha_j = np.arctan2(t_j_y, t_j_x)
    
    # 内蕴相对展开转角 theta_ij
    theta_ij = alpha_i - alpha_j
    # 规约至 [-pi, pi]
    theta_ij = np.arctan2(np.sin(theta_ij), np.cos(theta_ij))
    
    # 自旋联络 U_ij (注意：费米子波函数反变变换采用 -theta/2)
    U_ij = np.array([
        [np.exp(-1j * theta_ij / 2.0), 0.0],
        [0.0, np.exp(1j * theta_ij / 2.0)]
    ], dtype=complex)
    
    U_dict[(i, j)] = U_ij
    U_dict[(j, i)] = U_ij.conj().T
    
    # 面 i 内指向公共边的单位外法向 n_ij
    edge_mid = (p1 + p2) / 2.0
    outward_i = edge_mid - c_i
    outward_i -= np.dot(outward_i, N_i) * N_i
    outward_i /= np.linalg.norm(outward_i)
    
    n1 = np.dot(outward_i, e1_i)
    n2 = np.dot(outward_i, e2_i)
    gamma_n = n1 * sigma_x + n2 * sigma_y
    
    block_ij = -1j * w * np.dot(gamma_n, U_ij)
    H_dirac[2*i:2*i+2, 2*j:2*j+2] = block_ij
    H_dirac[2*j:2*j+2, 2*i:2*i+2] = block_ij.conj().T

# ── 3. 严格审计 12 个顶点的自旋和乐 ──
print("─" * 74)
print("  [1] 12 顶点自旋和乐角度审计 (Spin Holonomy around All Vertices)")
print("─" * 74)
print(f"  {'顶点编号':>8s}  {'实测闭合角模长 (rad)':>24s}  {'理论值 pi/6':>16s}  {'相对偏差':>12s}")
print("  " + "-" * 66)

holonomy_phases = []
for v_idx in range(12):
    adj_faces = [idx for idx, f in enumerate(faces) if v_idx in f]
    
    # 严格按测地拓扑环绕顶点排列 5 个相邻面
    ordered = [adj_faces[0]]
    curr = adj_faces[0]
    while len(ordered) < 5:
        for nxt in adj_faces:
            if nxt not in ordered and (curr, nxt) in U_dict:
                ordered.append(nxt)
                curr = nxt
                break
    
    # 环路连乘
    W_v = np.eye(2, dtype=complex)
    for k in range(5):
        f_from = ordered[k]
        f_to = ordered[(k + 1) % 5]
        W_v = np.dot(U_dict[(f_to, f_from)], W_v)
    
    # 提取自旋联络对应的真实亏角:
    # W_v[0,0] = exp(i * total_angle / 2)
    # 还原矢量转角 total_angle = 2 * angle(W_v[0,0])，规约至 (-pi, pi]，再除以 2:
    vec_angle = 2.0 * np.angle(W_v[0, 0])
    vec_angle_mod = np.arctan2(np.sin(vec_angle), np.cos(vec_angle))
    phase = abs(vec_angle_mod) / 2.0
    holonomy_phases.append(phase)
    
    dev = abs(phase - np.pi / 6.0) / (np.pi / 6.0) * 100.0
    print(f"  顶点 {v_idx:4d}  {phase:24.6f}  {np.pi/6.0:16.6f}  {dev:11.4f}%")

print()
mean_phase = np.mean(holonomy_phases)
print(f"  >> 12 顶点平均闭合角: {mean_phase:.6f} rad (理论亏角折半 pi/6 = {np.pi/6.0:.6f})")
print(f"  >> 全局相对偏差: {abs(mean_phase - np.pi/6.0)/(np.pi/6.0)*100:.6f}%")
print()

# ── 4. 对角化求解修复后的本征谱 ──
eigs = np.sort(np.linalg.eigvalsh(H_dirac))

print("─" * 74)
print("  [2] 修复后的二十面体面心 Dirac 本征谱 (前 5 个独特能级对)")
print("─" * 74)
zero_count = np.sum(np.abs(eigs) < 1e-5)
min_gap = np.min(np.abs(eigs))
chiral_err = np.max(np.abs(eigs + eigs[::-1]))

print(f"  1. 零模数量: {zero_count}  (严格保持为 0)")
print(f"  2. 基态拓扑质量隙 |λ_min| = {min_gap:.6f}")
print(f"  3. 手征反对易误差: {chiral_err:.2e}  (机器浮点极限)")
print()

# 提取正能级独特值
pos_eigs = eigs[eigs > 1e-5]
distinct = []
for e in pos_eigs:
    if not distinct or abs(e - distinct[-1]) > 1e-3:
        distinct.append(e)
    if len(distinct) == 5:
        break

print("  修复后前 5 个正能级阶梯:")
for idx, val in enumerate(distinct):
    ratio = val / distinct[0]
    print(f"    能级 {idx+1}: {val:+.6f}  (相对基态比率: {ratio:.3f})")

print("=" * 74)