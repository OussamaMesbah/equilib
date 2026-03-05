# API Reference

## Top-Level Function

### `solve_equilibrium(n_objs, subdivision=100, oracle=None)`
High-level factory function to find a topological fixed point.

- **Parameters:**
    - `n_objs` (int): Number of objectives to balance (>= 2).
    - `subdivision` (int, default=100): Resolution of the grid search (>= 2).
    - `oracle` (Callable, optional): A function taking a weight vector `w` (numpy array of shape `(n_objs,)`) and returning the index of the most dissatisfied objective (`0..n_objs-1`).
- **Returns:**
    - `np.ndarray` of optimal weights if `oracle` is provided.
    - `NDimEquilibSolver` instance if `oracle` is `None`.

---

## Core Classes

### `NDimEquilibSolver(n_objs, subdivision=100, device="cpu")`
PyTorch-native N-dimensional topological solver using implicit Kuhn/Freudenthal triangulation.

- **Parameters:**
    - `n_objs` (int): Number of objectives (>= 2).
    - `subdivision` (int): Grid resolution.
    - `device` (str): Torch device (`"cpu"` or `"cuda"`).
- **Methods:**
    - `solve(oracle_fn, batch_size=1)` → `torch.Tensor`: Runs the Sperner walk. `oracle_fn` receives a `(batch, n_objs)` weight tensor and returns a `(batch,)` long tensor of label indices.
    - `solve_generator()` → `Generator`: Interactive generator yielding `(vertex, weights, phase)` tuples and accepting label integers via `.send()`.
    - `get_barycentric_weights(y)` → `torch.Tensor`: Maps Kuhn lattice coordinates to simplex weights.

---

### `EquilibSolver(subdivision=10)`
Legacy 2D solver for 3-objective alignment problems.

- **Methods:**
    - `walk()` → `(triangle, path)`: Runs the Sperner walk on the 2-simplex. Returns the panchromatic triangle and the list of triangles visited.
    - `oracle_label(x, y)` → `int`: Returns the Sperner label at grid point `(x, y)`.
    - `weights_from_coords(x, y)` → `np.ndarray`: Converts grid coordinates to barycentric weights.

---

### `AdaptiveEquilibSolver(subdivision=10, max_depth=5, precision=1e-6)`
Iterative zoom-refinement solver for high-precision 3-objective alignment. Inherits from `EquilibSolver`.

- **Methods:**
    - `solve_adaptive()` → `List[np.ndarray] | None`: Runs iterative zoom and returns the three vertices of the final panchromatic triangle in global weight space.

---

### `SpernerTrainer(base_model, adapters, objectives, mock=True)`
Integration with Transformers/PEFT for LoRA weight merging.

- **Parameters:**
    - `base_model` (str or model): Hugging Face model identifier or model object.
    - `adapters` (List[str]): Names of LoRA adapters.
    - `objectives` (List[Callable]): Functions returning a scalar reward/loss.
    - `mock` (bool): If True (default), uses a synthetic loss landscape.
- **Methods:**
    - `train(grid_size=50)` → `np.ndarray`: Calculates optimal mixing weights.
    - `train_generator(grid_size=20)` → `Generator`: Interactive generator yielding `(weights, phase)` for human-in-the-loop feedback.
    - `evaluate_mixed_model(weights)` → `List[float]`: Returns objective losses for a weight mix.
    - `oracle_label(weights)` → `int`: Returns the index of the most dissatisfied objective.

---

### `NDimSurrogateEquilibSolver(n_objs, subdivision=50, n_init_samples=20, real_oracle=None)`
Active-learning surrogate solver that minimises expensive oracle calls using a KNN model.

- **Parameters:**
    - `real_oracle` (Callable, optional): `(weights: np.ndarray) → int`. Falls back to argmin mock if not provided.
    - `real_cost_delay` (float): Optional sleep simulating evaluation cost.
- **Methods:**
    - `solve_with_surrogate(max_iterations=15)` → `np.ndarray | None`: Iteratively refines the surrogate and returns optimal weights.

---

### `SurrogateEquilibSolver(subdivision=20, n_init_samples=10, real_cost_delay=0.1)`
Legacy 2D surrogate solver (3 objectives). Prefer `NDimSurrogateEquilibSolver` for new code.

---

### `AutoModelMerger(base_model_id, adapter_ids, device="cpu")`
Production-grade model merger. Finds the Nash equilibrium of conflicting capabilities.

- **Methods:**
    - `find_optimal_mix(evaluators, precision=50)` → `Dict[str, float]`: Returns a mapping of adapter names to optimal weights.

---

### `TopologicalMoERouter(num_experts, latent_dim=4096, device="cpu")`
Mixture-of-Experts router using topological equilibrium instead of softmax gating.

- **Methods:**
    - `forward_route(hidden_states, precision=20)` → `torch.Tensor`: Returns routing weights of shape `(num_experts,)`.

---

### `AgenticEquilibriumJudge(metrics, device="cpu")`
Automated alignment judge providing oracle labels using a simulated capability surface.

- **Methods:**
    - `get_labels(weights)` → `torch.Tensor`: Returns label indices for a batch of weight vectors.

---

### `auto_align_batch(n_objs, batch_size=128, device="cpu")`
Convenience function that creates an `AgenticEquilibriumJudge` and `NDimEquilibSolver`, then returns the solved equilibrium weights.

---

## Utilities

### `calculate_frustration_score(path_vertices)` → `float`
Measures topological frustration as the ratio of total path length to net displacement. Returns 1.0 for trivial paths, 999.0 for loops.

---

## Exceptions

### `SpernerConvergenceError`
Raised when the walk cannot find a panchromatic simplex due to improper labeling or boundary issues.
