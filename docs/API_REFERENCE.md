# API Reference

> **Contract for every solver:** the oracle you pass MUST satisfy the
> **Sperner boundary condition** — at any weight vector `w` with `w_i = 0`,
> the oracle must not return label `i`. Violations are silently rewritten by
> the solver; see [docs/THEORY.md § 3](THEORY.md#3-the-boundary-condition-and-silent-overrides).

## Top-level function

### `solve_equilibrium(n_objs, subdivision=100, oracle=None)`

Convenience factory wrapping `NDimEquilibSolver.solve`.

- **Parameters**
  - `n_objs` (int, ≥ 2): number of objectives. The simplex dimension is `n_objs - 1`.
  - `subdivision` (int, ≥ 2): grid resolution.
  - `oracle` (Callable, optional): `(w: np.ndarray of shape (n_objs,)) -> int`,
    returning the index of the most underserved objective at `w`. Must satisfy
    the Sperner boundary condition.
- **Returns**
  - `np.ndarray` of optimal weights if `oracle` is provided.
  - `NDimEquilibSolver` instance if `oracle` is `None`.

---

## Core classes

### `NDimEquilibSolver(n_objs, subdivision=100, device="cpu")`

PyTorch-native N-dimensional Sperner walk on an implicit Kuhn-Freudenthal triangulation.

- **Parameters**
  - `n_objs` (int, ≥ 2): number of objectives.
  - `subdivision` (int, ≥ 2): grid resolution.
  - `device` (str): `"cpu"` or `"cuda"`.
- **Methods**
  - `solve(oracle_fn, batch_size=1, max_restarts=3, random_start=False)` → `torch.Tensor`
    of shape `(batch_size, n_objs)`. Runs the dimension-lifting walk.
    `oracle_fn` receives a `(batch, n_objs)` weight tensor and returns a
    `(batch,)` long tensor of label indices.
  - `solve_generator()` → `typing.Generator[(vertex, weights, phase), int, np.ndarray]`:
    interactive coroutine — yields tuples and accepts a label via `.send(int)`.
    Returns the final centroid when the walk terminates.
  - `get_barycentric_weights(y)` → `torch.Tensor`: maps Kuhn lattice coordinates to simplex weights.

**Complexity:** `O(n_sub · d²)` pivot steps per walk; `d = n_objs - 1`.

---

### `EquilibSolver(subdivision=10, targets=None)`

Legacy explicit 2D Sperner walk for the 3-objective case.

- **Parameters**
  - `subdivision` (int): grid resolution.
  - `targets` (`np.ndarray` of shape `(3,)`, optional): the target weight
    vector used by the built-in `(w - target)²` synthetic oracle. Defaults to
    `[1/3, 1/3, 1/3]`. Subclass and override `oracle_label` for custom problems.
- **Methods**
  - `walk()` → `(triangle, path)`: runs the Sperner walk on the 2-simplex.
    Returns the panchromatic triangle and the path of triangles visited.
  - `oracle_label(x, y)` → `int`: returns the Sperner label at grid point `(x, y)`.
  - `weights_from_coords(x, y)` → `np.ndarray`: converts grid coordinates to barycentric weights.

---

### `AdaptiveEquilibSolver(subdivision=10, max_depth=5, precision=1e-6, targets=None)`

Iterative zoom-refinement of `EquilibSolver` for high-precision 3-objective walks.

- **Methods**
  - `solve_adaptive()` → `List[np.ndarray] | None`: runs iterative zoom and
    returns the three vertices of the final panchromatic triangle in global
    weight space.

---

### `SpernerTrainer(base_model, adapters, objectives, mock=True)`

Experimental Transformers/PEFT integration for LoRA weight merging.

- **Parameters**
  - `base_model` (str or model): Hugging Face model identifier or model object.
    Ignored in mock mode.
  - `adapters` (List[str]): names of LoRA adapters to blend.
  - `objectives` (List[Callable]): user-supplied scalar reward/loss functions.
  - `mock` (bool): if `True` (default), uses a synthetic loss landscape. If
    `False`, blends real adapter parameters and evaluates each objective on
    the blended model — emits a warning about cost.
- **Methods**
  - `train(grid_size=50)` → `np.ndarray`: returns the centroid of the panchromatic cell.
  - `train_generator(grid_size=20)` → `Generator`: interactive coroutine
    yielding `(weights, phase)` and accepting human labels via `.send(int)`.
  - `evaluate_mixed_model(weights)` → `List[float]`: per-objective scalars at `weights`.
  - `oracle_label(weights)` → `int`: index of the most underserved objective.

**Note:** in `mock=False` mode, every pivot triggers a full model evaluation. The
trainer caches by `(weights, 4-decimal)` to avoid duplicate calls, but expect
hundreds of evaluations on a 3-objective `grid_size=30` run.

---

### `NDimSurrogateEquilibSolver(n_objs, subdivision=50, n_init_samples=20, real_oracle=None)`

KNN active-learning wrapper for expensive oracles. Walks on a learned label
surrogate, verifies proposed fixed points against the real oracle, and
retrains on disagreements.

- **Parameters**
  - `real_oracle` (Callable, optional): `(weights: np.ndarray) -> int`. Falls
    back to argmin-mock if not provided. Must satisfy the Sperner boundary
    condition.
  - `real_cost_delay` (float): optional sleep simulating evaluation cost.
- **Methods**
  - `solve_with_surrogate(max_iterations=15)` → `np.ndarray | None`: iteratively
    refines the surrogate and returns the verified centroid.

**Note:** the panchromatic-cell guarantee does *not* hold under the surrogate
labeling — the KNN approximation may produce labels that aren't a valid
Sperner labeling. The verification step at the end checks the candidate
against the real oracle.

---

### `SurrogateEquilibSolver(subdivision=20, n_init_samples=10, real_cost_delay=0.1)`

Legacy 2D surrogate solver. Prefer `NDimSurrogateEquilibSolver`.

---

### `AutoModelMerger(base_model_id, adapter_ids, device="cpu")`

Thin wrapper that maps adapter names to simplex dimensions and runs an
`NDimEquilibSolver` with a user-supplied list of scalar evaluators.

- **Methods**
  - `find_optimal_mix(evaluators, precision=50)` → `Dict[str, float]`:
    returns `{adapter_name: weight}` for the panchromatic-cell centroid.

---

### `TopologicalMoERouter(num_experts, latent_dim=4096, device="cpu")`

**Research demo, not a production router.** Runs a full Sperner walk per
routed input. Slower than softmax routing by many orders of magnitude.

- **Methods**
  - `forward_route(hidden_states, precision=20)` → `torch.Tensor`:
    routing weights of shape `(num_experts,)` for the first token of the
    first batch element.

Instantiation emits a `UserWarning`. Use only for experimentation.

---

### `AgenticEquilibriumJudge(metrics, device="cpu")`

Helper for batch demos: a synthetic oracle that picks the most underserved
objective based on a built-in capability surface. Not a real model judge —
swap in a real reward model in production.

- **Methods**
  - `get_labels(weights)` → `torch.Tensor`: label indices for a batch of weights.

---

### `auto_align_batch(n_objs, batch_size=128, device="cpu")`

Convenience function: instantiates an `AgenticEquilibriumJudge` and runs a batch
of independent Sperner walks. Useful for stress-testing the solver, not for
production use.

---

## Utilities

### `calculate_frustration_score(path_vertices)` → `float`

Total path length divided by net displacement. Diagnostic of how much the
walk wandered — not a metric of solution quality. Returns `999.0` if the
walk returned to its start (a loop).

---

## Exceptions

### `SpernerConvergenceError`

Raised when the walk cannot find a panchromatic cell within its step budget.
Most commonly indicates an oracle that violates the Sperner boundary condition
in ways the silent override cannot rescue, or a `subdivision` too coarse for
the geometry of the labeling.
