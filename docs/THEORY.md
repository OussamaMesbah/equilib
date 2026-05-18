# Theory: Sperner's Lemma, KKM, and Brouwer

This document states precisely what the algorithm in this library does and
does not give you. It is deliberately conservative — earlier versions of the
docs conflated several distinct mathematical objects, and this rewrite
separates them.

## 1. The simplex

For `n ≥ 2` objectives, the **standard `(n-1)`-simplex** is

```
Δ = { w ∈ ℝⁿ  :  w_i ≥ 0  for all i,  Σ w_i = 1 }.
```

Each `w ∈ Δ` is a non-negative weighting of the `n` objectives summing to 1.
The vertices `V_0, …, V_{n-1}` of `Δ` are the standard basis vectors:
`V_i` has `w_i = 1` and all other coordinates 0.

The library represents `Δ` implicitly via a **Kuhn-Freudenthal triangulation**
at resolution `n_sub`. See `sperner/ndim_solver.py:get_barycentric_weights` for the
map between integer lattice coordinates and barycentric weights.

## 2. Sperner's Lemma (1928)

Let `T` be a triangulation of `Δ`. A **Sperner labeling** is an assignment
`L : (vertices of T) → {0, …, n-1}` satisfying:

1. `L(V_i) = i` for each main vertex.
2. **Boundary condition:** for any vertex `v` on the face `w_i = 0`,
   `L(v) ≠ i`.

A simplicial cell of `T` is **panchromatic** if its `n` vertices carry all `n`
labels.

> **Sperner's Lemma.** *Any Sperner-labeled triangulation of `Δ` contains an
> odd number of panchromatic cells. In particular, at least one exists.*

The library's `_run_walk` performs a constructive proof: it follows a path of
adjacent cells through the triangulation that is guaranteed to terminate at
a panchromatic cell. The centroid of that cell is the returned weight vector.

**This is the only mathematical guarantee the library provides.** Everything
below is conditional on the labeling actually being a Sperner labeling.

## 3. The boundary condition and silent overrides

The boundary condition `L(v) ≠ i` on the face `w_i = 0` is a **hard requirement** of
Sperner's Lemma. Without it, the lemma simply does not apply, and there is no
reason to expect a panchromatic cell to exist.

This library *silently rewrites* labels that violate the boundary condition.
The relevant code paths:

- [`sperner/ndim_solver.py:safe_oracle`](../sperner/ndim_solver.py): rows where the user's oracle
  returns label `i` on a point with `w_i ≤ 0` are replaced by the index of
  the largest nonzero coordinate.
- [`sperner/solver.py:oracle_label`](../sperner/solver.py): hardcoded `losses[i] = -1.0` when
  `w_i == 0`.
- [`sperner/moe_router.py:moe_oracle`](../sperner/moe_router.py): `starvation[w ≤ 1e-9] = -inf` before argmax.

The practical consequence: **if your oracle does not satisfy the boundary
condition, the walk will still terminate, but the returned centroid is
determined by the override heuristic, not by your oracle.** Make sure your
oracle handles the boundary case correctly. The recommended pattern is:

```python
def oracle(w):
    scores = my_metric(w)            # whatever your "underservedness" measure is
    scores[w <= 0] = -np.inf         # explicit Sperner boundary handling
    return int(np.argmax(scores))
```

## 4. From Sperner to KKM to Brouwer

Three related theorems are easy to confuse:

| Theorem | What it gives | When it applies |
| :--- | :--- | :--- |
| **Sperner's Lemma** | A panchromatic cell exists in any Sperner-labeled triangulation. | Pure combinatorics — no continuity needed. |
| **KKM (Knaster-Kuratowski-Mazurkiewicz)** | An intersection point of `n` closed sets that "cover" the simplex correctly. | A closed-set covering of `Δ` with the KKM property. |
| **Brouwer Fixed-Point Theorem** | A continuous map `f : Δ → Δ` has a fixed point. | `f` continuous. |

The three are equivalent in the sense that each implies the others. The
**centroid of a panchromatic cell** of a fine triangulation approximates a
fixed point of a continuous map `f : Δ → Δ` **only when the labeling is
induced by `f`** — for example, by the rule "label `w` with the index of the
largest deficit in `f(w) - w`".

If your oracle is a stochastic LLM judge, the panchromatic cell still exists
(as long as the boundary condition holds for whichever realisation you used at
each vertex), but the centroid no longer approximates the fixed point of any
canonical underlying map. The interpretation degrades from
"approximate Brouwer fixed point" to "centroid of a topologically distinguished
cell". This may still be useful, but the "fixed point" framing is misleading
when the labeling is noisy.

## 5. What this is *not*

### Not a Nash equilibrium

A Nash equilibrium is a fixed point of the **best-response correspondence** of
a strategic game `(I, S, u)` with players `I`, strategies `S`, and payoff
functions `u`. This library does not model players, strategies, or payoffs. The
panchromatic-cell centroid is a fixed point of an *abstract* map (when one
exists, see above), not the equilibrium of any game.

Earlier versions of the docs (and class docstrings) used the term "Nash
equilibrium" loosely. That was a category error and has been removed
throughout the library.

### Not a global optimum

The centroid of the panchromatic cell is the point at which all `n` Sperner
labels meet. There is no claim that it minimises any scalar objective or that
it sits on the Pareto frontier of the underlying objectives. In particular:

- If "argmin of objective scores" is your labeling rule, the panchromatic
  centroid is *not* the point minimising the worst objective. It is the
  point at which the "weakest objective" identity changes across all `n`
  alternatives.
- For Pareto-frontier exploration, use **NSGA-II** ([pymoo](https://pymoo.org/))
  or multi-objective Bayesian optimization (BoTorch's qNEHVI/ParEGO).

### Not "O(N) linear"

Sperner walks are **PPAD-complete** (Papadimitriou 1994): no polynomial-time
algorithm is known for finding panchromatic cells in arbitrary Sperner
labelings. The library's implementation bounds pivot steps per walk by

```
≲  MAX_PIVOT_STEPS_PER_CELL · n_sub · Σ_{k=1..d} (k + 1)
=  O(n_sub · d²)
```

(see `sperner/ndim_solver.py:_run_walk`). On well-behaved labelings the walk
usually terminates much earlier; on adversarial ones it can exhaust the budget.

## 6. Complexity

| Solver | Loop bound | Notes |
| :--- | :--- | :--- |
| `NDimEquilibSolver._run_walk` | `O(n_sub · d²)` per restart | `d = n_objs - 1`. |
| `EquilibSolver.walk` (legacy 2D) | `O(n_sub²)` | Hardcoded `max_steps = self.n * self.n * 2`. |
| `AdaptiveEquilibSolver.solve_adaptive` | `O(max_depth · n_sub²)` | Zoom refinement on 3 objectives. |
| `NDimSurrogateEquilibSolver` | Same as above × `max_iterations`, but most calls are routed to a KNN classifier rather than the real oracle. | Real-oracle count is empirical and depends on `n_init_samples`. |

For comparison, exhaustive grid search at resolution `X` over `n` objectives
costs `O(X^n)` oracle calls. Sperner is asymptotically better than grid
search, but **not** better than gradient methods (when gradients exist) or
multi-objective Bayesian optimization (which uses far fewer calls and
tolerates noise).

## 7. Recommended reading

- Sperner, E. (1928). *Neuer Beweis für die Invarianz der Dimensionszahl und des Gebietes*. Abh. Math. Sem. Hamburg, 6: 265–272.
- Scarf, H. (1967). *The Approximation of Fixed Points of a Continuous Mapping*. SIAM J. Appl. Math., 15(5): 1328–1343.
- Kuhn, H. W. (1968). *Simplicial approximation of fixed points*. PNAS, 61(4): 1238–1242.
- Freudenthal, H. (1942). *Simpliziale Zerlegungen von beschränkter Flachheit*. Annals of Mathematics, 43(3): 580–582.
- Papadimitriou, C. H. (1994). *On the complexity of the parity argument and other inefficient proofs of existence*. JCSS, 48(3): 498–532.
- Border, K. C. (1985). *Fixed Point Theorems with Applications to Economics and Game Theory*. Cambridge University Press. (For the relationship between Sperner, KKM, and Brouwer.)
