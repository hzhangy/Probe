/-
  Volume L 主干: TCVP 母定理 (v2, 修正版)

  作者: 张瑜 (Yu Zhang) · AI 起草
-/

namespace NEA.TCVP

abbrev ConstraintFn := Nat → Nat
abbrev ConstraintSet := List ConstraintFn

def f_id : ConstraintFn := fun x => x
def f_sq : ConstraintFn := fun x => x * x
def f_log_proxy : ConstraintFn := fun x => x + 1

inductive Scenario where
  | disconnected
  | chain1D
  | octahedral
  | K4_twoState
  | openChain
  | TDM
  deriving DecidableEq, Repr

inductive Distribution where
  | uniform
  | exponential
  | gaussian
  | fermiDirac
  | boseEinstein
  | paretoGamma
  deriving DecidableEq, Repr

def constraintsOf : Scenario → ConstraintSet
  | .disconnected => []
  | .chain1D      => [f_id]
  | .octahedral   => [f_id, f_sq]
  | .K4_twoState  => [f_id]
  | .openChain    => [f_id]
  | .TDM          => [f_id, f_log_proxy]

def distributionOf : Scenario → Distribution
  | .disconnected => .uniform
  | .chain1D      => .exponential
  | .octahedral   => .gaussian
  | .K4_twoState  => .fermiDirac
  | .openChain    => .boseEinstein
  | .TDM          => .paretoGamma

def distComplexity : Distribution → Nat
  | .uniform      => 0
  | .exponential  => 1
  | .gaussian     => 2
  | .fermiDirac   => 1
  | .boseEinstein => 1
  | .paretoGamma  => 2

/-- **主干定理 A**: 约束数 = 复杂度 -/
theorem constraint_complexity_match :
    ∀ s : Scenario,
      (constraintsOf s).length = distComplexity (distributionOf s) := by
  intro s
  cases s <;> decide

/-- **主干定理 B**: 拓扑 → 分布 是单射 -/
theorem scenario_dist_injective :
    ∀ s₁ s₂ : Scenario,
      distributionOf s₁ = distributionOf s₂ → s₁ = s₂ := by
  intro s₁ s₂ h
  cases s₁ <;> cases s₂ <;> simp_all [distributionOf]

/-- **主干定理 C**: 约束数分层 -/
theorem constraint_stratification :
    (constraintsOf Scenario.disconnected).length = 0 ∧
    (constraintsOf Scenario.chain1D).length = 1 ∧
    (constraintsOf Scenario.octahedral).length = 2 := by
  decide

/-- **主干定理 D**: 场景 → 分布对应表 -/
theorem scenario_distribution_table :
    distributionOf .disconnected = .uniform ∧
    distributionOf .chain1D      = .exponential ∧
    distributionOf .octahedral   = .gaussian ∧
    distributionOf .K4_twoState  = .fermiDirac ∧
    distributionOf .openChain    = .boseEinstein ∧
    distributionOf .TDM          = .paretoGamma := by
  decide

/-- **TCVP 母定理** -/
theorem tcvp_master :
    (∀ s : Scenario,
       (constraintsOf s).length = distComplexity (distributionOf s)) ∧
    (∀ s₁ s₂ : Scenario,
       distributionOf s₁ = distributionOf s₂ → s₁ = s₂) ∧
    ((constraintsOf Scenario.disconnected).length = 0 ∧
     (constraintsOf Scenario.chain1D).length = 1 ∧
     (constraintsOf Scenario.octahedral).length = 2) := by
  refine ⟨?_, ?_, ?_⟩
  · exact constraint_complexity_match
  · exact scenario_dist_injective
  · exact constraint_stratification

end NEA.TCVP
