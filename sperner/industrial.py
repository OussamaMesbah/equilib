import logging
from typing import Callable, Dict, List

import numpy as np
import torch

from .ndim_solver import NDimEquilibSolver

logger = logging.getLogger(__name__)


class AutoModelMerger:
    """Thin wrapper that maps adapter names to simplex dimensions and runs
    an :class:`NDimEquilibSolver` against a user-supplied list of scalar
    evaluators.

    Returns the centroid of the panchromatic cell found by the walk, as a
    ``{adapter_name: weight}`` dictionary. This is a convenience wrapper —
    the actual adapter blending is left to the caller, since merging
    semantics vary by framework (PEFT, mergekit, custom).

    **Oracle contract.** Each evaluator is a callable ``(weights) -> float``.
    The wrapper builds the Sperner oracle by picking the index of the
    objective with the largest gap to the per-call maximum. Make sure your
    evaluators return finite values on the simplex boundary; otherwise the
    Sperner boundary condition may be violated.

    Args:
        base_model_id: Identifier for the base model (e.g. HuggingFace repo id).
            Stored for reference; not loaded here.
        adapter_ids: List of adapter/capability identifiers to merge.
        device: Torch device for the underlying solver.
    """

    def __init__(
        self,
        base_model_id: str,
        adapter_ids: List[str],
        device: str = "cpu",
    ) -> None:
        self.base_model_id = base_model_id
        self.adapter_ids = adapter_ids
        self.capability_names = [aid.split('/')[-1] for aid in adapter_ids]
        self.device = device

    def find_optimal_mix(self,
                         evaluators: List[Callable],
                         precision: int = 50) -> Dict[str, float]:
        """Run the Sperner walk and return the centroid mapping.

        Args:
            evaluators: Per-objective ``(weights) -> float`` callables. The
                wrapper picks the objective with the largest gap to the
                per-call maximum as the Sperner label.
            precision: ``subdivision`` parameter passed to the underlying solver.

        Returns:
            ``{adapter_name: weight}`` dict summing to ~1.
        """
        logger.info(
            f"Starting AutoModelMerger walk for: {self.capability_names}")

        solver = NDimEquilibSolver(n_objs=len(self.adapter_ids),
                                   subdivision=precision,
                                   device=self.device)

        def industrial_oracle(weights_batch: torch.Tensor) -> torch.Tensor:
            """
            Vectorized oracle for the industrial merger.
            """
            batch_size = weights_batch.shape[0]
            labels = torch.zeros(batch_size,
                                 dtype=torch.long,
                                 device=self.device)

            for i in range(batch_size):
                weights = weights_batch[i].cpu().numpy()
                scores = []
                for ev in evaluators:
                    scores.append(ev(weights))

                # We want to find the objective with the largest gap to the max score
                # (The most dissatisfied capability)
                scores_np = np.array(scores)
                labels[i] = int(np.argmax(np.max(scores_np) - scores_np))

            return labels

        # High-performance synchronous solve
        best_weights_tensor = solver.solve(oracle_fn=industrial_oracle,
                                           batch_size=1)

        # Convert back to dictionary
        best_weights = best_weights_tensor[0].cpu().numpy()
        return dict(zip(self.capability_names, best_weights))


def run_enterprise_demo():
    print("=" * 50)
    print("AutoModelMerger demo (synthetic evaluators)")
    print("=" * 50)

    merger = AutoModelMerger(
        "meta-llama/Llama-3",
        ["adapters/speed", "adapters/accuracy", "adapters/safety"])

    # Synthetic linear evaluators — purely illustrative.
    def speed_eval(w):
        return float(w[0] * 0.9)

    def accuracy_eval(w):
        return float(w[1] * 0.95 - (w[0] * 0.1))

    def safety_eval(w):
        return float(w[2] * 1.0 - (w[1] * 0.2))

    evaluators = [speed_eval, accuracy_eval, safety_eval]

    print("Running Sperner walk over 3 synthetic evaluators...")
    result = merger.find_optimal_mix(evaluators)

    print("\nPanchromatic-cell centroid:")
    for cap, weight in result.items():
        print(f"  {cap:10}: {weight*100:5.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_enterprise_demo()
