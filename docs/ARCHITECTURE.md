# Architecture

## Core modules

### 1. `NDimEquilibSolver` (`sperner.ndim_solver`)

The core N-dimensional Sperner walk. Implements:

- **Implicit Kuhn-Freudenthal triangulation** — neighbours are computed via
  algebraic pivots on lattice coordinates, so memory is `O(d)` rather than
  `O(n_sub^d)`.
- **Dimension lifting** — the walk warms up on the boundary `d=1` face
  (an edge), then `d=2` (a triangle), ..., then the full `d`-simplex. This
  helps the walk reach the interior faster in practice.
- **Optional multi-start** — `solve(max_restarts=...)` retries from
  randomised interior starting points if the first walk lands on or near a
  boundary face.

The loop bound is `O(n_sub · d²)` pivots; see
[docs/THEORY.md § Complexity](THEORY.md#6-complexity).

### 2. `EquilibSolver` (`sperner.solver`)

Legacy 2D solver for the 3-objective case. Kept for readability and for
backward compatibility with the adaptive zoom solver. The 2D triangulation is
explicit, so the code is easier to trace.

The legacy solver's oracle target is now a constructor argument
(`targets=None` defaults to centred `[1/3, 1/3, 1/3]`), so the same class can
be reused without subclassing.

### 3. `AdaptiveEquilibSolver` (`sperner.adaptive_solver`)

Iterative-refinement ("zoom") wrapper around the 3-objective solver: after
finding a panchromatic triangle, it re-bases its coordinate system onto that
triangle and runs again. With `max_depth = D` zoom levels, the achievable
precision is `O(n_sub^(-D))`. Only works for 3 objectives because it relies on
the explicit 2D triangulation.

### 4. `NDimSurrogateEquilibSolver` (`sperner.surrogate_solver`)

Wraps the `NDimEquilibSolver` walk with a **KNN active-learning surrogate**:
the walk queries the surrogate rather than the real oracle, then verifies the
surrogate's proposed fixed point against the real oracle and retrains on
disagreements. Cuts real-oracle calls from `O(n_sub · d²)` to ~20–50 in
practice, at the cost of (a) no longer satisfying the Sperner panchromatic
guarantee (the KNN labels are an approximation), and (b) requiring the user
to trust the convergence test in `solve_with_surrogate`.

### 5. `SpernerTrainer` (`sperner.sperner_trainer`)

Integration helper for PEFT/LoRA adapter mixing. In `mock=True` mode it uses
a synthetic loss landscape (no model loading). In `mock=False` mode it blends
adapter parameters according to the candidate weights and calls user-supplied
objective functions on the blended model — this is **expensive** and emits a
warning. See the trainer's docstring for the cost analysis.

### 6. `TopologicalMoERouter` (`sperner.moe_router`)

A research demo, not a production routing layer. Running a full Sperner walk
per token is many orders of magnitude slower than softmax routing. Kept in
the library for educational purposes; the class emits a warning on
instantiation and at most one walk is run per call.

### 7. Human UI (`sperner.human_ui`)

Streamlit interface for manual labeling — at each step the user picks the
objective they think is currently weakest. Educational tool.

### 8. Utilities

- `analytics.calculate_frustration_score` — total walk path length divided
  by net displacement. A diagnostic, not a metric of solution quality.
- `plotting.plot_simplex_heatmap` / `plot_sperner_path` — visualizations
  for 3-objective walks only.

## Data flow (typical synchronous call)

1. User instantiates a solver with `n_objs`, `subdivision`, and optionally `device`.
2. User defines an `oracle_fn(weights_batch) -> labels_batch` satisfying the
   Sperner boundary condition.
3. `solver.solve()` runs the dimension-lifting walk, calling `oracle_fn` at
   each new vertex.
4. When the walk finds a panchromatic cell, the centroid is returned.
5. (Optional) On a restart, the walk starts from a randomised interior point
   if the first attempt converged too close to a boundary face.

## What the library does **not** do

- Train models (no gradient computation, no parameter updates).
- Provide objective implementations (the oracle is user-supplied).
- Compute Pareto fronts.
- Solve Nash equilibria (no game model).
- Handle noisy oracles (the walk assumes deterministic labeling).
