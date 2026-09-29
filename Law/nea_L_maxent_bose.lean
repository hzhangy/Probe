import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic

set_option linter.unusedSectionVars false

namespace NEA.MaxEnt.Bose

open Finset

variable {ι : Type*} [Fintype ι]

noncomputable def boseEntropy (n : ℝ) : ℝ :=
  (1 + n) * Real.log (1 + n) - n * Real.log n

noncomputable def boseKL (n nstar : ℝ) : ℝ :=
  n * Real.log (n / nstar) - (1 + n) * Real.log ((1 + n) / (1 + nstar))

/-- Bose 相对熵非负。数学事实，Lean 证明通过凹函数切线留给 v2。 -/
theorem bose_kl_nonneg
    (n nstar : ℝ) (hn : 0 < n) (hns : 0 < nstar) :
    0 ≤ boseKL n nstar := by
  sorry

/-- Bose 熵差恒等式 -/
theorem boseEntropy_diff
    (n nstar : ℝ) (hn : 0 < n) (hns : 0 < nstar) :
    boseEntropy nstar - boseEntropy n =
      boseKL n nstar + (nstar - n) * Real.log ((1 + nstar) / nstar) := by
  unfold boseEntropy boseKL
  have hn_ne : n ≠ 0 := ne_of_gt hn
  have hns_ne : nstar ≠ 0 := ne_of_gt hns
  have h1n_ne : 1 + n ≠ 0 := ne_of_gt (by linarith : 0 < 1 + n)
  have h1ns_ne : 1 + nstar ≠ 0 := ne_of_gt (by linarith : 0 < 1 + nstar)
  rw [Real.log_div hn_ne hns_ne, Real.log_div h1n_ne h1ns_ne,
      Real.log_div h1ns_ne hns_ne]
  ring

theorem bose_maxent
    (E : ι → ℝ) (n nstar : ι → ℝ)
    (hn : ∀ i, 0 < n i) (hns : ∀ i, 0 < nstar i)
    (β α : ℝ)
    (hns_log : ∀ i, Real.log ((1 + nstar i) / nstar i) = β * E i + α)
    (h_N : ∑ i, n i = ∑ i, nstar i)
    (h_E : ∑ i, n i * E i = ∑ i, nstar i * E i) :
    ∑ i, boseEntropy (n i) ≤ ∑ i, boseEntropy (nstar i) := by
  have h_each : ∀ i, boseEntropy (nstar i) - boseEntropy (n i) =
      boseKL (n i) (nstar i) + (nstar i - n i) * (β * E i + α) := by
    intro i
    have h := boseEntropy_diff (n i) (nstar i) (hn i) (hns i)
    rw [hns_log i] at h
    linarith
  -- 约束消去: Σ (nstar - n) = 0
  have h_N_zero : ∑ i, (nstar i - n i) = 0 := by
    rw [Finset.sum_sub_distrib]
    exact sub_eq_zero.mpr h_N.symm
  -- 约束消去: Σ (nstar - n) · E = 0
  have h_E_zero : ∑ i, (nstar i - n i) * E i = 0 := by
    have h2 : ∀ i, (nstar i - n i) * E i =
        nstar i * E i - n i * E i := fun i => by ring
    simp_rw [h2]
    rw [Finset.sum_sub_distrib]
    exact sub_eq_zero.mpr h_E.symm
  -- 关键: Σ (nstar - n) (β E + α) = 0
  have h_extract : ∑ i, (nstar i - n i) * (β * E i + α) = 0 := by
    calc ∑ i, (nstar i - n i) * (β * E i + α)
        = ∑ i, (β * ((nstar i - n i) * E i)
                + α * (nstar i - n i)) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
      _ = β * (∑ i, (nstar i - n i) * E i)
          + α * (∑ i, (nstar i - n i)) := by
          rw [Finset.sum_add_distrib,
              ← Finset.mul_sum, ← Finset.mul_sum]
      _ = β * 0 + α * 0 := by rw [h_E_zero, h_N_zero]
      _ = 0 := by ring
  have h_sum : ∑ i, (boseEntropy (nstar i) - boseEntropy (n i)) =
      ∑ i, boseKL (n i) (nstar i) := by
    calc ∑ i, (boseEntropy (nstar i) - boseEntropy (n i))
        = ∑ i, (boseKL (n i) (nstar i)
                + (nstar i - n i) * (β * E i + α)) := by
            apply Finset.sum_congr rfl
            intro i _; exact h_each i
      _ = ∑ i, boseKL (n i) (nstar i)
          + ∑ i, (nstar i - n i) * (β * E i + α) := by
            rw [Finset.sum_add_distrib]
      _ = ∑ i, boseKL (n i) (nstar i) := by
            rw [h_extract]; ring
  have h_gibbs : 0 ≤ ∑ i, boseKL (n i) (nstar i) := by
    apply Finset.sum_nonneg
    intro i _
    exact bose_kl_nonneg (n i) (nstar i) (hn i) (hns i)
  have h_diff : ∑ i, (boseEntropy (nstar i) - boseEntropy (n i)) =
      ∑ i, boseEntropy (nstar i) - ∑ i, boseEntropy (n i) := by
    rw [Finset.sum_sub_distrib]
  linarith [h_sum, h_gibbs, h_diff]

end NEA.MaxEnt.Bose
