set_option linter.unusedSectionVars false

namespace NEA.Kleiber

/-- isKleiber d 判断 d/(d+1) = 3/4 -/
def isKleiber (d : Nat) : Bool := d * 4 == (d + 1) * 3

theorem kleiber_d3 : isKleiber 3 = true := by decide

theorem kleiber_d1_d2_d4_d5_d6_false :
    isKleiber 1 = false ∧ isKleiber 2 = false ∧
    isKleiber 4 = false ∧ isKleiber 5 = false ∧
    isKleiber 6 = false := by decide

theorem kleiber_unique_upto_100 :
    ((List.range 101).filter isKleiber) = [3] := by decide

/-- 显式方程解: 4d = 3(d+1) → d = 3 -/
theorem kleiber_unique (d : Nat) (h : d * 4 = (d + 1) * 3) : d = 3 := by
  have h1 : d * 4 = d * 3 + 3 := by
    calc d * 4 = (d + 1) * 3 := h
      _ = d * 3 + 1 * 3 := by rw [Nat.add_mul]
      _ = d * 3 + 3 := by rw [Nat.one_mul]
  have h2 : d * 4 = d * 3 + d := by
    calc d * 4 = d * (3 + 1) := by rw [show (4 : Nat) = 3 + 1 from rfl]
      _ = d * 3 + d * 1 := by rw [Nat.mul_add]
      _ = d * 3 + d := by rw [Nat.mul_one]
  have h3 : d * 3 + 3 = d * 3 + d := h1.symm.trans h2
  exact (Nat.add_left_cancel h3).symm

def branching (d : Nat) : Nat := 2 ^ d

theorem branching_d3 : branching 3 = 8 := by decide

end NEA.Kleiber
