import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic

namespace NEA.MaxEnt

set_option linter.unusedSectionVars false

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Shannon 熵 S[p] = -Σ p_i log p_i -/
noncomputable def entropy (p : ι → ℝ) : ℝ :=
  -∑ i, p i * Real.log (p i)

/-- KL 散度 D(p || q) = Σ p_i log(p_i / q_i) -/
noncomputable def klDiv (p q : ι → ℝ) : ℝ :=
  ∑ i, p i * Real.log (p i / q i)

/-- **Gibbs 不等式**: D(p || q) ≥ 0 对概率分布 p, q。

    证明: 用 Real.log_le_sub_one_of_pos: log x ≤ x - 1。
    逐项: p_i - q_i ≤ p_i log(p_i / q_i), 求和得 0 ≤ D(p||q)。
-/
theorem gibbs_inequality
    (p q : ι → ℝ)
    (hp_pos : ∀ i, 0 < p i)
    (hq_pos : ∀ i, 0 < q i)
    (hp_sum : ∑ i, p i = 1)
    (hq_sum : ∑ i, q i = 1) :
    0 ≤ klDiv p q := by
  unfold klDiv
  have h : ∀ i ∈ (Finset.univ : Finset ι),
      p i - q i ≤ p i * Real.log (p i / q i) := by
    intro i _
    have hp := hp_pos i
    have hq := hq_pos i
    have hqp : 0 < q i / p i := div_pos hq hp
    have h1 : Real.log (q i / p i) ≤ q i / p i - 1 :=
      Real.log_le_sub_one_of_pos hqp
    have h2 : p i * Real.log (q i / p i) ≤ p i * (q i / p i - 1) :=
      mul_le_mul_of_nonneg_left h1 (le_of_lt hp)
    have h3 : p i * (q i / p i - 1) = q i - p i := by field_simp
    have h4 : p i * Real.log (q i / p i) ≤ q i - p i := by linarith
    have h5 : p i * Real.log (p i / q i) = -p i * Real.log (q i / p i) := by
      have hp_ne : p i ≠ 0 := ne_of_gt hp
      have hq_ne : q i ≠ 0 := ne_of_gt hq
      rw [show p i / q i = (q i / p i)⁻¹ by field_simp]
      rw [Real.log_inv]
      ring
    nlinarith
  have hsum : ∑ i, (p i - q i) ≤ ∑ i, p i * Real.log (p i / q i) :=
    Finset.sum_le_sum h
  rw [Finset.sum_sub_distrib, hp_sum, hq_sum] at hsum
  linarith

/-- 辅助: 若 Z = Σ exp(-β f_i), 则 Σ exp(-β f_i) / Z = 1 -/
private theorem exp_sum_div_eq_one
    (f : ι → ℝ) (β Z : ℝ) (hZ_pos : 0 < Z)
    (hZ_def : Z = ∑ i, Real.exp (-β * f i)) :
    ∑ i, Real.exp (-β * f i) / Z = 1 := by
  simp_rw [div_eq_mul_inv]
  rw [← Finset.sum_mul, ← hZ_def]
  exact mul_inv_cancel₀ (ne_of_gt hZ_pos)

/-- **辅助引理**: 指数族分布的 KL 散度等于熵差。

    设 p*_i = exp(-β f_i) / Z, 且 Σ p_i f_i = Σ p*_i f_i = c。
    则 D(p || p*) = S[p*] - S[p]。
-/
theorem klDiv_exp_eq_entropy_diff
    (p : ι → ℝ) (hp_pos : ∀ i, 0 < p i) (hp_sum : ∑ i, p i = 1)
    (f : ι → ℝ) (c β Z : ℝ) (hZ_pos : 0 < Z)
    (hZ_def : Z = ∑ i, Real.exp (-β * f i))
    (hp_constraint : ∑ i, p i * f i = c)
    (hps_constraint : ∑ i, Real.exp (-β * f i) / Z * f i = c) :
    klDiv p (fun i => Real.exp (-β * f i) / Z) =
      entropy (fun i => Real.exp (-β * f i) / Z) - entropy p := by
  have hpstar_pos : ∀ i, 0 < Real.exp (-β * f i) / Z := fun i =>
    div_pos (Real.exp_pos _) hZ_pos
  have hpstar_sum : ∑ i, Real.exp (-β * f i) / Z = 1 :=
    exp_sum_div_eq_one f β Z hZ_pos hZ_def
  -- log(p*_i) = -β f_i - log Z
  have hlog : ∀ i, Real.log (Real.exp (-β * f i) / Z) =
      -β * f i - Real.log Z := by
    intro i
    rw [Real.log_div (ne_of_gt (Real.exp_pos _)) (ne_of_gt hZ_pos)]
    rw [Real.log_exp]
  -- 计算 Σ p log p*
  have h_plogpstar :
      ∑ i, p i * Real.log (Real.exp (-β * f i) / Z) =
      -β * c - Real.log Z := by
    calc ∑ i, p i * Real.log (Real.exp (-β * f i) / Z)
        = ∑ i, p i * (-β * f i - Real.log Z) := by
            apply Finset.sum_congr rfl
            intro i _
            rw [hlog i]
      _ = ∑ i, (-β * (p i * f i) - Real.log Z * p i) := by
            apply Finset.sum_congr rfl
            intro i _
            ring
      _ = ∑ i, (-β * (p i * f i)) - ∑ i, (Real.log Z * p i) := by
            rw [Finset.sum_sub_distrib]
      _ = -β * (∑ i, p i * f i) - Real.log Z * (∑ i, p i) := by
            rw [← Finset.mul_sum, ← Finset.mul_sum]
      _ = -β * c - Real.log Z * 1 := by rw [hp_constraint, hp_sum]
      _ = -β * c - Real.log Z := by ring
  -- 计算 Σ p* log p*
  have h_pstarlogpstar :
      ∑ i, Real.exp (-β * f i) / Z * Real.log (Real.exp (-β * f i) / Z) =
      -β * c - Real.log Z := by
    calc ∑ i, Real.exp (-β * f i) / Z * Real.log (Real.exp (-β * f i) / Z)
        = ∑ i, Real.exp (-β * f i) / Z * (-β * f i - Real.log Z) := by
            apply Finset.sum_congr rfl
            intro i _
            rw [hlog i]
      _ = ∑ i, (-β * (Real.exp (-β * f i) / Z * f i)
                - Real.log Z * (Real.exp (-β * f i) / Z)) := by
            apply Finset.sum_congr rfl
            intro i _
            ring
      _ = ∑ i, (-β * (Real.exp (-β * f i) / Z * f i))
          - ∑ i, (Real.log Z * (Real.exp (-β * f i) / Z)) := by
            rw [Finset.sum_sub_distrib]
      _ = -β * (∑ i, Real.exp (-β * f i) / Z * f i)
          - Real.log Z * (∑ i, Real.exp (-β * f i) / Z) := by
            rw [← Finset.mul_sum, ← Finset.mul_sum]
      _ = -β * c - Real.log Z * 1 := by rw [hps_constraint, hpstar_sum]
      _ = -β * c - Real.log Z := by ring
  -- 展开 KL, 用 sum_sub_distrib 分离
  unfold klDiv entropy
  have h_split :
      ∑ i, p i * Real.log (p i / (Real.exp (-β * f i) / Z)) =
      ∑ i, p i * Real.log (p i)
        - ∑ i, p i * Real.log (Real.exp (-β * f i) / Z) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    rw [Real.log_div (ne_of_gt (hp_pos i)) (ne_of_gt (hpstar_pos i))]
    ring
  rw [h_split]
  linarith

/-- **主定理**: MaxEnt 解是指数族。

    给定约束 ⟨f⟩ = c, 设 β 使 p*_i = exp(-β f_i)/Z 满足该约束。
    则对任意满足约束的概率分布 p, S[p] ≤ S[p*]。
-/
theorem maxent_is_exp_family
    (f : ι → ℝ) (c β Z : ℝ) (hZ_pos : 0 < Z)
    (hZ_def : Z = ∑ i, Real.exp (-β * f i))
    (h_constraint_star : ∑ i, Real.exp (-β * f i) / Z * f i = c)
    (p : ι → ℝ) (hp_pos : ∀ i, 0 < p i) (hp_sum : ∑ i, p i = 1)
    (hp_constraint : ∑ i, p i * f i = c) :
    entropy p ≤ entropy (fun i => Real.exp (-β * f i) / Z) := by
  have hpstar_pos : ∀ i, 0 < Real.exp (-β * f i) / Z := fun i =>
    div_pos (Real.exp_pos _) hZ_pos
  have hpstar_sum : ∑ i, Real.exp (-β * f i) / Z = 1 :=
    exp_sum_div_eq_one f β Z hZ_pos hZ_def
  have h_gibbs : 0 ≤ klDiv p (fun i => Real.exp (-β * f i) / Z) :=
    gibbs_inequality p (fun i => Real.exp (-β * f i) / Z)
      hp_pos hpstar_pos hp_sum hpstar_sum
  have h_eq : klDiv p (fun i => Real.exp (-β * f i) / Z) =
      entropy (fun i => Real.exp (-β * f i) / Z) - entropy p :=
    klDiv_exp_eq_entropy_diff p hp_pos hp_sum f c β Z hZ_pos hZ_def
      hp_constraint h_constraint_star
  linarith

end NEA.MaxEnt
