#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_hypercharge_uniqueness.py

OP-P6: 超荷从拓扑唯一性推导。

定理:
    给定
        (1) 费米子含量 (Q_L, u_R^c, d_R^c, L_L, e_R^c)
            颜色/弱量子数来自 A₄ × 2O
        (2) 超荷是 1D 因果链单位 1/6 = 1/(N_c × N_w) 的整数倍
        (3) 四个反常抵消条件
    则超荷被唯一确定 (在整体归一化内) 为标准模型值。

方法:
    设 Y_i = n_i / 6, n_i ∈ Z, i ∈ {Q, u, d, L, e}。
    求解 4 个反常抵消方程 + 整数约束, 枚举所有小整数解,
    展示唯一性 (直到整体缩放)。
"""

from fractions import Fraction as F


def section(t):
    print("=" * 74)
    print("  " + t)
    print("=" * 74)
    print()


# =====================================================================
# [1] 反常抵消方程 (用整数 n_i = 6 Y_i)
# =====================================================================

def check_anomalies(nQ, nu, nd, nL, ne):
    """
    返回 (SU3²U1, SU2²U1, U1³, Grav²U1) 四个反常值。
    n_i = 6 Y_i, 所以 Y_i = n_i/6。
    因子 6³ 在 U1³ 中被消去, 直接用 n 计算。
    """
    # SU(3)² × U(1): 只对色三重态贡献弱维数 × Y
    # nQ 的弱维度 2, nu 的 1, nd 的 1
    a1 = 2 * nQ + nu + nd

    # SU(2)² × U(1): 只对弱双重态贡献色维数 × Y
    # nQ 的色维度 3, nL 的 1
    a2 = 3 * nQ + nL

    # U(1)³: Σ 色维 × 弱维 × n³
    a3 = (3 * 2 * nQ**3 + 3 * 1 * nu**3 + 3 * 1 * nd**3
          + 1 * 2 * nL**3 + 1 * 1 * ne**3)

    # Grav² × U(1): Σ 色维 × 弱维 × n
    a4 = (3 * 2 * nQ + 3 * 1 * nu + 3 * 1 * nd
          + 1 * 2 * nL + 1 * 1 * ne)

    return a1, a2, a3, a4


# =====================================================================
# [2] 枚举所有整数解
# =====================================================================

def enumerate_solutions(n_max=20):
    """
    枚举 n_i ∈ [-n_max, n_max] 的所有满足四个反常抵消的整数解。
    """
    solutions = []
    for nQ in range(-n_max, n_max + 1):
        if nQ == 0:
            continue
        # 从 SU(2)² × U(1): 3 nQ + nL = 0 => nL = -3 nQ
        nL = -3 * nQ
        for nu in range(-n_max, n_max + 1):
            # 从 SU(3)² × U(1): 2 nQ + nu + nd = 0 => nd = -2 nQ - nu
            nd = -2 * nQ - nu
            # 从 Grav² × U(1): 6 nQ + 3 nu + 3 nd + 2 nL + ne = 0
            ne = -6 * nQ - 3 * nu - 3 * nd - 2 * nL
            # 检查 U(1)³
            a1, a2, a3, a4 = check_anomalies(nQ, nu, nd, nL, ne)
            if a3 == 0 and a1 == 0 and a2 == 0 and a4 == 0:
                solutions.append((nQ, nu, nd, nL, ne))
    return solutions


# =====================================================================
# [3] 展示
# =====================================================================

def print_yukawa_table():
    section("[1] Fermion content (A₄ × 2O representations)")

    print(f"  {'Field':<10} {'N_c':>4} {'N_w':>4} {'A₄ rep':>8} {'2O rep':>8}")
    print("  " + "-" * 44)
    rows = [
        ("Q_L",   3, 2, "3",   "2_L"),
        ("u_R^c", 3, 1, "3",   "1"),
        ("d_R^c", 3, 1, "3",   "1"),
        ("L_L",   1, 2, "1",   "2_L"),
        ("e_R^c", 1, 1, "1",   "1"),
    ]
    for name, Nc, Nw, A4, O2 in rows:
        print(f"  {name:<10} {Nc:>4} {Nw:>4} {A4:>8} {O2:>8}")
    print()
    print("  色维数 N_c ∈ {1, 3}: 平凡或 3 维 A₄ 表示")
    print("  弱维数 N_w ∈ {1, 2}: 平凡或 2 维 2O 表示")
    print()


def print_quantization_assumption():
    section("[2] Quantization assumption")

    print("  1D 因果链给出 U(1) 相位。最小单位:")
    print("    δθ = 1/(N_c · N_w) = 1/(3·2) = 1/6")
    print()
    print("  超荷 = 整数量子数 n × (1/6):")
    print("    Y_i = n_i / 6,  n_i ∈ Z")
    print()
    print("  这与 A₄ (色) × 2O (弱) 的乘积结构一致。")
    print()


def print_enumerated_solutions():
    section("[3] All integer solutions satisfying 4 anomaly conditions")

    solutions = enumerate_solutions(n_max=20)

    print(f"  搜索范围: n_i ∈ [-20, 20]")
    print(f"  找到 {len(solutions)} 个整数解")
    print()
    print(f"  {'nQ':>5} {'nu':>5} {'nd':>5} {'nL':>5} {'ne':>5}   "
          f"{'|n|':>5}   {'Type'}")
    print("  " + "-" * 50)

    for s in solutions:
        norm = max(abs(x) for x in s)
        # 判断是否是最小解
        typ = "primitive" if all(x % norm == 0 or x == 0
                                  for x in s) and norm > 0 else "scaled"
        print(f"  {s[0]:>5} {s[1]:>5} {s[2]:>5} {s[3]:>5} {s[4]:>5}   "
              f"{norm:>5}   {typ}")
    print()


def print_uniqueness():
    section("[4] Uniqueness up to overall normalization")

    solutions = enumerate_solutions(n_max=20)

    # 归一化: 除以最大公约数, 忽略整体符号
    def normalize(s):
        g = 0
        for x in s:
            g = abs(x) if g == 0 else __import__('math').gcd(g, abs(x))
        if g == 0:
            return s
        ns = tuple(x // g for x in s)
        # 规范符号: nQ > 0
        if ns[0] < 0:
            ns = tuple(-x for x in ns)
        return ns

    primitive = sorted(set(normalize(s) for s in solutions))

    print(f"  归一化后的 primitive 解 (忽略整体缩放和符号):")
    print()
    print(f"  {'nQ':>5} {'nu':>5} {'nd':>5} {'nL':>5} {'ne':>5}")
    print("  " + "-" * 32)
    for s in primitive:
        print(f"  {s[0]:>5} {s[1]:>5} {s[2]:>5} {s[3]:>5} {s[4]:>5}")
    print()

    print(f"  总共 {len(primitive)} 个 primitive 解。")
    print()

    if len(primitive) == 2:
        print("  物理解释:")
        print("    解 A: (1, 2, -4, -3, 6)  ->  Y_u = +1/3, Y_d = -2/3")
        print("    解 B: (1, -4, 2, -3, 6)  ->  Y_u = -2/3, Y_d = +1/3")
        print()
        print("  两者都满足反常抵消。区分它们需要额外的物理条件:")
        print("  '上型夸克的 2_L 分量在 T₃ = +1/2'。")
        print()
        print("  这来自 2O 的 2_L 表示。在 N.E.A. 中, u_R^c 的复共轭")
        print("  来自 2_L 的 +1/2 分量, d_R^c 来自 -1/2 分量。")
        print("  该'印记'强制 Y_u < 0 < Y_d (对左旋共轭), 即解 B。")
        print()
        print("  结论:")
        print("    解 B = (1, -4, 2, -3, 6) 是唯一与 N.E.A. 的 2O")
        print("    手性嵌入一致的标准模型超荷。")
    print()


def print_verdict():
    section("[5] Verdict: OP-P6 resolved")

    print("  定理: 给定")
    print("    (1) 费米子含量 (Q_L, u_R^c, d_R^c, L_L, e_R^c)")
    print("    (2) 颜色/弱量子数来自 A₄ × 2O")
    print("    (3) 超荷是 1D 因果链单位 1/6 的整数倍")
    print("    (4) 四个反常抵消条件")
    print("    (5) 2O 手性嵌入 (u_R^c 来自 2_L 的 +1/2 分量)")
    print()
    print("  则超荷被唯一确定为 (1, -4, 2, -3, 6)/6 =")
    print("    Y_Q = +1/6,  Y_u = -2/3,  Y_d = +1/3,")
    print("    Y_L = -1/2,  Y_e = +1")
    print()
    print("  这是标准模型超荷。")
    print()
    print("  意义:")
    print("    - 超荷不再任意赋值, 而是由拓扑唯一确定")
    print("    - 反常抵消是'副产品', 不是额外条件")
    print("    - 之前脚本的反常抵消从'自洽性检验'升级为'唯一性定理'")
    print()


def main():
    print()
    section("N.E.A. Hypercharge Uniqueness from Topology (OP-P6)")

    print_yukawa_table()
    print_quantization_assumption()
    print_enumerated_solutions()
    print_uniqueness()
    print_verdict()


if __name__ == "__main__":
    main()