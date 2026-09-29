import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic

set_option linter.unusedSectionVars false

namespace NEA.MaxEnt.Fermi

open Finset

variable {ι : Type*} [Fintype ι]

noncomputable def binEntropy (p : ℝ) : ℝ :=
  -p * Real.log p - (1 - p) * Real.log (1 - p)

noncomputable def binKL (p q : ℝ) : ℝ :=
  p * Real.log (p / q) + (1 - p) * Real.log ((1 - p) / (1 - q))

theorem binary_gibbs
    (p q : ℝ) (hp : 0 < p) (hp1 : p < 1) (hq : 0 < q) (hq1 : q < 1) :
    0 ≤ binKL p q := by
  have hp_ne : p ≠ 0 := ne_of_gt hp
  have hq_ne : q ≠ 0 := ne_of_gt hq
  have hp1_pos : 0 < 1 - p := by linarith
  have hq1_pos : 0 < 1 - q := by linarith
  have hlog_pq : Real.log (p / q) ≥ 1 - q / p := by
    have h1 : Real.log (q / p) ≤ q / p - 1 :=
      Real.log_le_sub_one_of_pos (div_pos hq hp)
    have h2 : Real.log (p / q) = -Real.log (q / p) := by
      rw [show p / q = (q / p)⁻¹ by field_simp]
      rw [Real.log_inv]
    linarith
  have hlog_p1q1 :
      Real.log ((1 - p) / (1 - q)) ≥ 1 - (1 - q) / (1 - p) := by
    have h1 : Real.log ((1 - q) / (1 - p)) ≤ (1 - q) / (1 - p) - 1 :=
      Real.log_le_sub_one_of_pos (div_pos hq1_pos hp1_pos)
    have h2 : Real.log ((1 - p) / (1 - q)) =
        -Real.log ((1 - q) / (1 - p)) := by
      rw [show (1 - p) / (1 - q) = ((1 - q) / (1 - p))⁻¹ by field_simp]
      rw [Real.log_inv]
    linarith
  have h1 : p * Real.log (p / q) ≥ p - q := by
    have hmul := mul_le_mul_of_nonneg_left hlog_pq (le_of_lt hp)
    have hrw : p * (1 - q / p) = p - q := by field_simp
    linarith
  have h2 : (1 - p) * Real.log ((1 - p) / (1 - q)) ≥ q - p := by
    have hmul := mul_le_mul_of_nonneg_left hlog_p1q1 (le_of_lt hp1_pos)
    have hrw : (1 - p) * (1 - (1 - q) / (1 - p)) = q - p := by
      field_simp; ring
    linarith
  unfold binKL
  linarith

theorem binEntropy_diff
    (p q : ℝ) (hp : 0 < p) (hp1 : p < 1) (hq : 0 < q) (hq1 : q < 1) :
    binEntropy q - binEntropy p =
      binKL p q + (p - q) * Real.log (q / (1 - q)) := by
  unfold binEntropy binKL
  have hp_ne : p ≠ 0 := ne_of_gt hp
  have hq_ne : q ≠ 0 := ne_of_gt hq
  have hp1_ne : 1 - p ≠ 0 := ne_of_gt (by linarith : 0 < 1 - p)
  have hq1_ne : 1 - q ≠ 0 := ne_of_gt (by linarith : 0 < 1 - q)
  rw [Real.log_div hp_ne hq_ne, Real.log_div hp1_ne hq1_ne,
      Real.log_div hq_ne hq1_ne]
  ring

theorem fermi_maxent
    (E : ι → ℝ) (p pstar : ι → ℝ)
    (hp : ∀ i, 0 < p i) (hp1 : ∀ i, p i < 1)
    (hps : ∀ i, 0 < pstar i) (hps1 : ∀ i, pstar i < 1)
    (β α : ℝ)
    (hps_logit : ∀ i, Real.log (pstar i / (1 - pstar i)) = -(β * E i + α))
    (h_N : ∑ i, p i = ∑ i, pstar i)
    (h_E : ∑ i, p i * E i = ∑ i, pstar i * E i) :
    ∑ i, binEntropy (p i) ≤ ∑ i, binEntropy (pstar i) := by
  have h_each : ∀ i, binEntropy (pstar i) - binEntropy (p i) =
      binKL (p i) (pstar i) + (pstar i - p i) * (β * E i + α) := by
    intro i
    have h := binEntropy_diff (p i) (pstar i)
      (hp i) (hp1 i) (hps i) (hps1 i)
    rw [hps_logit i] at h
    have hrw : (p i - pstar i) * (-(β * E i + α)) =
        (pstar i - p i) * (β * E i + α) := by ring
    linarith
  have h_N_zero : ∑ i, (pstar i - p i) = 0 := by
    rw [Finset.sum_sub_distrib]
    exact sub_eq_zero.mpr h_N.symm
  have h_E_zero : ∑ i, (pstar i - p i) * E i = 0 := by
    have h2 : ∀ i, (pstar i - p i) * E i =
        pstar i * E i - p i * E i := fun i => by ring
    simp_rw [h2]
    rw [Finset.sum_sub_distrib]
    exact sub_eq_zero.mpr h_E.symm
  have h_extract : ∑ i, (pstar i - p i) * (β * E i + α) = 0 := by
    calc ∑ i, (pstar i - p i) * (β * E i + α)
        = ∑ i, (β * ((pstar i - p i) * E i) + α * (pstar i - p i)) := by
          apply Finset.sum_congr rfl; intro i _; ring
      _ = β * (∑ i, (pstar i - p i) * E i)
          + α * (∑ i, (pstar i - p i)) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      _ = β * 0 + α * 0 := by rw [h_E_zero, h_N_zero]
      _ = 0 := by ring
  have h_sum : ∑ i, (binEntropy (pstar i) - binEntropy (p i)) =
      ∑ i, binKL (p i) (pstar i) := by
    calc ∑ i, (binEntropy (pstar i) - binEntropy (p i))
        = ∑ i, (binKL (p i) (pstar i) +
                (pstar i - p i) * (β * E i + α)) := by
            apply Finset.sum_congr rfl
            intro i _; exact h_each i
      _ = ∑ i, binKL (p i) (pstar i)
          + ∑ i, (pstar i - p i) * (β * E i + α) := by
            rw [Finset.sum_add_distrib]
      _ = ∑ i, binKL (p i) (pstar i) := by rw [h_extract]; ring
  have h_gibbs : 0 ≤ ∑ i, binKL (p i) (pstar i) := by
    apply Finset.sum_nonneg
    intro i _
    exact binary_gibbs (p i) (pstar i) (hp i) (hp1 i) (hps i) (hps1 i)
  have h_diff : ∑ i, (binEntropy (pstar i) - binEntropy (p i)) =
      ∑ i, binEntropy (pstar i) - ∑ i, binEntropy (p i) := by
    rw [Finset.sum_sub_distrib]
  linarith [h_sum, h_gibbs, h_diff]

end NEA.MaxEnt.Fermi
