"""Research demo: per-token MoE routing via a Sperner walk.

**This module is not a production routing layer.** Running a full Sperner walk
per routed input is many orders of magnitude slower than softmax routing, and
the routing-collapse problem that motivates this demo has much cheaper fixes
(load-balancing loss, expert dropout, top-k routing). The class is kept in the
library only for educational and benchmarking purposes; instantiation emits a
:class:`UserWarning`.

For a meaningful routing decision, the underlying ``hidden_state -> expert``
mapping should be learned (e.g. a small MLP trained with a balancing
objective). Sperner is not a substitute for that.
"""

import logging
import warnings

import torch

from .ndim_solver import NDimEquilibSolver

logger = logging.getLogger(__name__)


class TopologicalMoERouter:
    """Per-token MoE routing weights via a Sperner walk (research demo).

    Maps a hidden state to a vector on the ``(num_experts - 1)``-simplex by
    running a Sperner walk against the projection ``Linear(latent_dim,
    num_experts)`` applied to the input. The walk finds the centroid of a
    panchromatic cell of the labeling

        label(w) = argmax_i  (projection[i] - w[i])    s.t.  w[i] > 0

    Args:
        num_experts: Number of experts (>= 2).
        latent_dim: Hidden-state dimension (default 4096).
        device: Torch device.
    """

    def __init__(self,
                 num_experts: int,
                 latent_dim: int = 4096,
                 device: str = "cpu") -> None:
        if num_experts < 2:
            raise ValueError(
                f"num_experts must be >= 2, got {num_experts}")
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.device = device
        self.routing_proj = torch.nn.Linear(latent_dim, num_experts)
        warnings.warn(
            "TopologicalMoERouter runs a full Sperner walk per call. It is a "
            "research demo, not a production router — softmax / top-k routing "
            "is many orders of magnitude faster. See the module docstring.",
            UserWarning,
            stacklevel=2,
        )
        logger.info(
            f"Initialized TopologicalMoERouter for {num_experts} experts "
            f"(research demo).")

    def forward_route(self,
                      hidden_states: torch.Tensor,
                      precision: int = 20) -> torch.Tensor:
        """Compute routing weights for the first token of the first batch element.

        Args:
            hidden_states: Tensor of shape ``(batch, seq_len, latent_dim)``.
            precision: Grid resolution for the Sperner walk.

        Returns:
            Tensor of shape ``(num_experts,)`` summing to ~1.

        Note:
            Only the ``[0, 0, :]`` slice of ``hidden_states`` is used. To route
            multiple tokens, call this method in a loop — but expect each call
            to be much slower than a softmax gate.
        """
        expert_dissatisfaction = self.routing_proj(hidden_states)
        token_dissatisfaction = expert_dissatisfaction[0, 0].detach()

        solver = NDimEquilibSolver(n_objs=self.num_experts,
                                   subdivision=precision,
                                   device=self.device)

        def moe_oracle(weights_batch: torch.Tensor) -> torch.Tensor:
            # Vectorised: starvation[i] = projection[i] - w[i], with w[i] <= 0
            # forced to -inf so the Sperner boundary condition holds.
            bs = weights_batch.shape[0]
            starvation = token_dissatisfaction.unsqueeze(0).expand(
                bs, -1) - weights_batch
            starvation = torch.where(weights_batch > 1e-9, starvation,
                                     torch.full_like(starvation,
                                                     -float('inf')))
            return starvation.argmax(dim=-1)

        optimal_routing_weights = solver.solve(oracle_fn=moe_oracle,
                                               batch_size=1)
        return optimal_routing_weights[0].clone().detach()


def run_moe_demo():
    """Print a short demo run of the topological router.

    This is wrapped in __main__ to keep the demo separate from the importable
    class. Output is deliberately plain (no emoji, no marketing claims).
    """
    print("=" * 60)
    print("TopologicalMoERouter — research demo")
    print("=" * 60)

    num_experts = 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        router = TopologicalMoERouter(num_experts=num_experts)

    mock_hidden_state = torch.randn(1, 1, 4096)
    print(f"Running Sperner walk over {num_experts} experts "
          f"(this is slow on purpose — see module docstring)...")
    optimal_weights = router.forward_route(mock_hidden_state, precision=30)

    print("Routing weights:")
    for i, w in enumerate(optimal_weights.numpy().tolist()):
        if w > 0.01:
            print(f"  Expert {i}: {w:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run_moe_demo()
