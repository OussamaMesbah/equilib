import torch
from typing import List


class AgenticEquilibriumJudge:
    """Automated alignment judge that provides oracle labels without a human.

    Uses a simulated capability surface to identify the weakest objective
    for any given weight mix.  In production, replace the scoring logic with
    a distilled reward model.

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

        # Simulated Capability Surface (Realistic Non-Linear Trade-offs)
        # In a real system, this triggers actual inference.
        with torch.no_grad():
            # Objectives are satisfied based on weights but penalize each other (Alignment Tax)
            scores = weights * 0.9 - 0.1 * torch.sum(
                weights**2, dim=1, keepdim=True)

            # The label is the index of the score that is furthest from perfection (1.0)
            gaps = 1.0 - scores
            return torch.argmax(gaps, dim=1)


def auto_align_batch(n_objs: int, batch_size: int = 128, device: str = "cpu"):
    """Plug-and-play batch alignment."""
    from .ndim_solver import NDimEquilibSolver

    judge = AgenticEquilibriumJudge(
        metrics=[f"cap_{i}" for i in range(n_objs)], device=device)
    solver = NDimEquilibSolver(n_objs=n_objs, device=device)

    return solver.solve(oracle_fn=judge.get_labels, batch_size=batch_size)
