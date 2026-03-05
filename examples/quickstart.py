"""Sperner Quickstart — Find optimal weights for 3 conflicting objectives."""

import numpy as np
import torch
from sperner import NDimEquilibSolver


def simulate_llm_eval(weights: np.ndarray) -> np.ndarray:
    """Mock LLM evaluator returning [Helpfulness, Safety, Verbosity] scores."""
    w_h, w_s, w_v = weights
    score_h = np.clip(0.8 * w_h + 0.2 * w_v - 0.3 * w_s + 0.1, 0, 1)
    score_s = np.clip(0.9 * w_s - 0.1 * w_h + 0.1, 0, 1)
    score_v = np.clip(0.6 * w_v + 0.4 * w_h - 0.2 * w_s, 0, 1)
    return np.array([score_h, score_s, score_v])


def main():
    targets = np.array([0.7, 0.8, 0.4])
    print(
        f"Target: Helpfulness={targets[0]}, Safety={targets[1]}, Verbosity={targets[2]}"
    )

    solver = NDimEquilibSolver(n_objs=3, subdivision=50)

    def judge(w_batch: torch.Tensor) -> torch.Tensor:
        labels = []
        for i in range(w_batch.shape[0]):
            w = w_batch[i].cpu().numpy()
            gaps = targets - simulate_llm_eval(w)
            labels.append(int(np.argmax(gaps)))
        return torch.tensor(labels, dtype=torch.long)

    result = solver.solve(oracle_fn=judge, batch_size=1)
    optimal = result[0].cpu().numpy()
    final = simulate_llm_eval(optimal)

    print(
        f"\nOptimal Weights:  H={optimal[0]:.3f}, S={optimal[1]:.3f}, V={optimal[2]:.3f}"
    )
    print(
        f"Final Metrics:    H={final[0]:.3f}, S={final[1]:.3f}, V={final[2]:.3f}"
    )
    print(
        f"Targets:          H={targets[0]:.3f}, S={targets[1]:.3f}, V={targets[2]:.3f}"
    )


if __name__ == "__main__":
    main()
