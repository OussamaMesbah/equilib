"""Example: KNN active-learning surrogate for expensive oracles.

The surrogate solver wraps the core Sperner walk with a KNN model that
learns oracle labels from a small number of real evaluations. Real-oracle
calls typically drop from O(n_sub * d^2) to ~20-50, at the cost of losing
the Sperner panchromatic guarantee (the KNN labels are an approximation).
A verification step against the real oracle is run before the centroid is
returned. See ``docs/THEORY.md`` for caveats.
"""

import numpy as np
from sperner import NDimSurrogateEquilibSolver


def expensive_evaluator(weights: np.ndarray) -> int:
    """Synthetic 'expensive' oracle.

    Returns the index of the most underserved objective using a fixed-target
    scoring rule. The Sperner boundary condition is enforced manually.
    In a real workflow, replace this with a benchmark-suite call.
    """
    targets = np.array([0.6, 0.7, 0.5, 0.4])
    # Deterministic version — the original demo used random noise, which
    # violates Sperner's combinatorial assumptions and makes the walk's
    # convergence test unreliable.
    scores = weights * targets
    gaps = targets - scores
    gaps[weights <= 0] = -np.inf  # Sperner boundary condition
    return int(np.argmax(gaps))


def main():
    objectives = ["Helpfulness", "Safety", "Creativity", "Conciseness"]
    print(f"Objectives: {objectives}")

    solver = NDimSurrogateEquilibSolver(
        n_objs=4,
        subdivision=30,
        n_init_samples=15,
        real_oracle=expensive_evaluator,
    )

    result = solver.solve_with_surrogate(max_iterations=15)
    if result is None:
        print("Surrogate did not converge — try increasing max_iterations "
              "or n_init_samples.")
        return

    print(f"\nCentroid: {np.round(result, 3)}")
    print(f"Real oracle calls used: {solver.real_queries}")
    print(f"(Underlying walk bound would have been ~"
          f"{30 * 3 * 3} pivots before the surrogate.)")


if __name__ == "__main__":
    main()
