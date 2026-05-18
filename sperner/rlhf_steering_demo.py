"""RLHF-style steering demo using a synthetic reward-model surface.

This is a *demonstration script*, not part of the public API. It models a
toy 3-objective trade-off (Helpfulness, Safety, Verbosity) using a hand-
crafted linear "reward surface" and runs an NDimEquilibSolver walk against
the gap-to-target labeling.

The synthetic surface is intentionally simple; the script is meant to
illustrate how to wire a domain-specific oracle into the solver, not to
make any claim about real RLHF. Real RLHF tunes parameters from preference
signals — this is not that.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RLHFSteeringOracle:
    """Synthetic reward-model surface for the steering demo."""

    def evaluate_model(self, w_h: float, w_s: float, w_v: float) -> np.ndarray:
        """Return the simulated [Helpfulness, Safety, Verbosity] scores for
        the given mixing weights.

        Each score is a hand-crafted linear function of the weights, clipped
        to ``[0, 1]``. The trade-offs are designed so that no single corner of
        the simplex maximises all three.
        """
        total = w_h + w_s + w_v
        if total == 0:
            return np.array([0.0, 0.0, 0.0])
        h, s, v = w_h / total, w_s / total, w_v / total

        score_h = 0.8 * h + 0.2 * v - 0.3 * s + 0.1
        score_s = 0.9 * s - 0.1 * h + 0.1
        score_v = 0.6 * v + 0.4 * h - 0.2 * s

        return np.clip([score_h, score_s, score_v], 0, 1)


from sperner.ndim_solver import NDimEquilibSolver

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("RLHF steering demo (3 objectives, synthetic reward surface)")

    oracle = RLHFSteeringOracle()
    targets = np.array([0.7, 0.8, 0.4])
    solver = NDimEquilibSolver(n_objs=3, subdivision=30)

    def rlhf_label(weights_batch: torch.Tensor) -> torch.Tensor:
        """Sperner oracle: index of the objective with the largest deficit
        relative to its target. Sperner boundary condition enforced
        explicitly."""
        batch_size = weights_batch.shape[0]
        labels = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            w = weights_batch[i].numpy()
            metrics = oracle.evaluate_model(w[0], w[1], w[2])
            gaps = targets - metrics
            gaps[w <= 1e-9] = -np.inf  # Sperner boundary
            labels[i] = int(np.argmax(gaps))
        return labels

    best_w = solver.solve(oracle_fn=rlhf_label,
                         batch_size=1,
                         max_restarts=5,
                         random_start=True)[0].numpy()

    logger.info("=" * 40)
    logger.info("Centroid (panchromatic-cell mid-point):")
    logger.info(f"  Helpfulness weight: {best_w[0]:.3f}")
    logger.info(f"  Safety weight:      {best_w[1]:.3f}")
    logger.info(f"  Verbosity weight:   {best_w[2]:.3f}")

    final_out = oracle.evaluate_model(*best_w)
    logger.info("-" * 40)
    logger.info("Predicted scores at the centroid:")
    logger.info(f"  Helpfulness: {final_out[0]:.3f} (target {targets[0]})")
    logger.info(f"  Safety:      {final_out[1]:.3f} (target {targets[1]})")
    logger.info(f"  Verbosity:   {final_out[2]:.3f} (target {targets[2]})")
    logger.info("=" * 40)
