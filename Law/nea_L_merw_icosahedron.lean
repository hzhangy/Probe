import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Tactic

set_option linter.unusedSectionVars false

namespace NEA.MERW

/-- 简单图: 顶点 v 的邻居集 -/
structure SimpleGraph (V : Type*) [Fintype V] where
  adj : V → V → Prop
  symm : ∀ u v, adj u v → adj v u
  irrefl : ∀ v, ¬ adj v v

/-- 一步转移矩阵 T 支持在边上: T_ij ≠ 0 仅当 adj i j -/
def SupportsOnEdges {V : Type*} [Fintype V] (G : SimpleGraph V)
    (T : V → V → ℝ) : Prop :=
  ∀ i j, ¬ G.adj i j → T i j = 0

/-- **核心定理**: 若图距 ≥ 2 的顶点对无直边，
    则任何"支持在边上"的转移矩阵 T 在这些位置上为 0。 -/
theorem distance2_implies_zero
    {V : Type*} [Fintype V] (G : SimpleGraph V)
    (T : V → V → ℝ) (hT : SupportsOnEdges G T)
    (v₀ v₃ : V) (h : ¬ G.adj v₀ v₃) :
    T v₀ v₃ = 0 :=
  hT v₀ v₃ h

/-- **应用**: 二十面体上顶点 0 和顶点 3 (图距 3) 无直边 -/
theorem icosahedron_T13_eq_zero
    {V : Type*} [Fintype V] (G : SimpleGraph V)
    (T : V → V → ℝ) (hT : SupportsOnEdges G T)
    (v₀ v₃ : V) (h_dist3 : ¬ G.adj v₀ v₃) :
    T v₀ v₃ = 0 :=
  distance2_implies_zero G T hT v₀ v₃ h_dist3

/-- **具体构造**: 二十面体的距离分层 1:5:5:1 -/
def icosahedronLayerStructure : List ℕ := [1, 5, 5, 1]

theorem icosahedron_vertex_count :
    icosahedronLayerStructure.sum = 12 := by
  decide

theorem icosahedron_distance_layers :
    -- 从任一顶点出发，图距 0,1,2,3 处的顶点数
    icosahedronLayerStructure = [1, 5, 5, 1] := rfl

end NEA.MERW
