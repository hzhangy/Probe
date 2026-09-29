set_option linter.unusedSectionVars false

namespace NEA.Dim3

/-- J(d) > 0 ⟺ d ≥ 3 且 d(d-1) < 8 -/
def isSolvent (d : Nat) : Bool :=
  decide (3 ≤ d) && decide (d * (d - 1) < 8)

/-- d = 3 是 solvent -/
theorem d3_solvent : isSolvent 3 = true := by decide

/-- d = 4 不是 solvent -/
theorem d4_insolvent : isSolvent 4 = false := by decide

/-- 穷举: d ∈ [0, 100] 中唯一 solvent 是 d = 3 -/
theorem d3_unique_upto_100 :
    ((List.range 101).filter isSolvent) = [3] := by decide

/-- 一般定理: d ≥ 4 则 d(d-1) ≥ 12 > 8 -/
theorem d_ge_4_insolvent (d : Nat) (hd : 4 ≤ d) : d * (d - 1) > 8 := by
  have h1 : 3 ≤ d - 1 := Nat.sub_le_sub_right hd 1
  have h2 : 4 * 3 ≤ d * (d - 1) := Nat.mul_le_mul hd h1
  omega

end NEA.Dim3
