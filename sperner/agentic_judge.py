import torch
from typing import List


class AgenticEquilibriumJudge:
    """Synthetic-oracle helper for batch demos.

    Returns the index of the most underserved objective using a simple linear
    capability surface ``scores = w * 0.9 - 0.1 * sum(w^2)``. This is a
    smoke-test oracle, not a real evaluator — in any non-demo use the scoring
    rule should be replaced with a domain-specific metric (e.g. a benchmark
    score, a distilled reward model).

    Args:
        metrics: Human-readable names for each objective dimension.
        device: Torch device for tensor operations.
    """

    def __init__(self, metrics: List[str], device: str = "cpu") -> None:
        self.metrics = metrics
        self.device = device

    def get_labels(self, weights: torch.Tensor) -> torch.Tensor:
        """Return the index of the weakest objective for each row.

        Args:
            weights: Tensor of shape ``(batch, n_objs)`` with non-negative weights.

        Returns:
            Long tensor of shape ``(batch,)`` with label indices.
        """
        batch_size = weights.shape[0]
        n_objs = weights.shape[1]

        # Synthetic linear capability surface. Real-world oracles should
        # replace this with an actual evaluation function.
        with torch.no_grad():
            scores = weights * 0.9 - 0.1 * torch.sum(
                weights**2, dim=1, keepdim=True)
            gaps = 1.0 - scores
            # Sperner boundary: do not return label i where w_i = 0.
            gaps = torch.where(weights > 0, gaps,
                               torch.full_like(gaps, -float('inf')))
            return torch.argmax(gaps, dim=1)


def auto_align_batch(n_objs: int, batch_size: int = 128, device: str = "cpu"):
    """Run a batch of independent Sperner walks against the synthetic judge.

    Useful for stress-testing the solver. Not a real alignment pipeline —
    the judge is a smoke-test oracle.
    """
    from .ndim_solver import NDimEquilibSolver

    judge = AgenticEquilibriumJudge(
        metrics=[f"cap_{i}" for i in range(n_objs)], device=device)
    solver = NDimEquilibSolver(n_objs=n_objs, device=device)

    return solver.solve(oracle_fn=judge.get_labels, batch_size=batch_size)
