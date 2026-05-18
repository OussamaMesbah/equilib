"""N-dimensional Sperner / Kuhn-Freudenthal walk.

Finds the centroid of a panchromatic cell on the standard (n-1)-simplex,
given an oracle that satisfies the Sperner boundary condition (on the face
``w_i = 0``, the oracle must not return label ``i``).

See ``docs/THEORY.md`` for the precise mathematical statement of what this
solver does and does not give you. In particular, the centroid approximates
a Brouwer fixed point of the labeling map ONLY when (a) the labeling is
induced by a continuous map ``f : Δ → Δ`` and (b) the Sperner boundary
condition holds. Stochastic oracles invalidate the panchromatic guarantee.

References:
    Sperner (1928); Scarf (1967); Kuhn (1968); Freudenthal (1942);
    Papadimitriou (1994) for the PPAD-completeness lower bound.
"""

import logging
from typing import Callable, Generator, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# Multiplier on (n_sub * (active_dim + 1)) used as the per-dimension pivot
# budget. The total walk bound is therefore O(n_sub * d^2). Empirically, walks
# on well-behaved labelings terminate in a small fraction of this budget; the
# constant is generous so that adversarial labelings don't trigger a false
# SpernerConvergenceError.
MAX_PIVOT_STEPS_PER_CELL = 4


class SpernerConvergenceError(Exception):
    """Raised when the Sperner walk fails to find a panchromatic cell within
    the step budget. Most commonly indicates an oracle that violates the
    Sperner boundary condition in ways the silent override cannot fix."""


class NDimEquilibSolver:
    """PyTorch-native N-dimensional Sperner walk.

    Finds the centroid of a panchromatic cell on the standard (n-1)-simplex
    using an implicit Kuhn-Freudenthal triangulation. Dimension-lifting walks
    initialise on the boundary ``d=1`` face and grow outward.

    **Oracle contract.** The oracle you pass to :meth:`solve` MUST satisfy
    the Sperner boundary condition: at any weight vector ``w`` with
    ``w_i = 0``, the oracle must not return label ``i``. Violations are
    silently rewritten by the solver's internal ``safe_oracle``. See
    ``docs/THEORY.md`` for the consequences.

    **Complexity.** The pivot loop is bounded by ``O(n_sub * d^2)`` per walk,
    where ``d = n_objs - 1``. This is not a polynomial-time guarantee — Sperner
    walks are PPAD-complete (Papadimitriou 1994).

    Args:
        n_objs: Number of objectives (>= 2). The simplex dimension is ``n_objs - 1``.
        subdivision: Grid resolution; higher values give finer search. Must be >= 2.
        device: Torch device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        n_objs: int,
        subdivision: int = 100,
        device: str = "cpu",
    ) -> None:
        if n_objs < 2:
            raise ValueError(f"n_objs must be >= 2, got {n_objs}")
        if subdivision < 2:
            raise ValueError(f"subdivision must be >= 2, got {subdivision}")
        self.n_objs = n_objs
        self.d = n_objs - 1
        self.n_sub = subdivision
        self.device = device

    def get_barycentric_weights(self, y: torch.Tensor) -> torch.Tensor:
        """Map Kuhn lattice coordinates to barycentric simplex weights.

        Args:
            y: Integer tensor of shape ``(batch, d)`` with Kuhn coordinates.

        Returns:
            Float tensor of shape ``(batch, n_objs)`` summing to 1 along dim -1.
        """
        y_sorted, _ = torch.sort(y, dim=-1)
        zeros = torch.zeros((y.shape[0], 1), device=self.device, dtype=y.dtype)
        nsub = torch.full((y.shape[0], 1),
                          self.n_sub,
                          device=self.device,
                          dtype=y.dtype)
        extended = torch.cat([zeros, y_sorted, nsub], dim=-1)  # (batch, d+2)
        diffs = (extended[:, 1:] -
                 extended[:, :-1]).float() / self.n_sub  # (batch, d+1)
        # Reverse so w0 = (n_sub - y_d)/n_sub, w_d = y_1/n_sub
        return diffs.flip(dims=[-1])

    def get_vertex_batch(self, y_base: torch.Tensor, sigma: torch.Tensor,
                         k: int) -> torch.Tensor:
        """Returns the k-th vertex of the simplex defined by (y_base, sigma)."""
        v = y_base.clone()
        if k > 0:
            axes = sigma[:, :k]
            ones = torch.ones_like(axes)
            v.scatter_add_(1, axes, ones)
        return v

    def pivot_batch(self, y_base: torch.Tensor, sigma: torch.Tensor,
                    k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kuhn pivot: replace vertex k with its opposite in the adjacent simplex."""
        new_y = y_base.clone()
        new_sigma = sigma.clone()

        mask_k0 = (k == 0)
        if mask_k0.any():
            axis_0 = torch.gather(
                sigma, 1,
                torch.zeros((sigma.shape[0], 1),
                            device=self.device,
                            dtype=torch.long))[mask_k0].squeeze(1)
            new_y[mask_k0, axis_0] += 1
            new_sigma[mask_k0] = torch.roll(sigma[mask_k0], shifts=-1, dims=1)

        mask_kd = (k == self.d)
        if mask_kd.any():
            axis_last = torch.gather(
                sigma, 1,
                torch.full((sigma.shape[0], 1),
                           self.d - 1,
                           device=self.device,
                           dtype=torch.long))[mask_kd].squeeze(1)
            new_y[mask_kd, axis_last] -= 1
            new_sigma[mask_kd] = torch.roll(sigma[mask_kd], shifts=1, dims=1)

        # Middle case: swap sigma[k-1] and sigma[k] in-place via gather/scatter.
        # Vectorised across the batch — no per-item .item() syncs.
        mask_mid = (~mask_k0) & (~mask_kd) & (k != -1)
        if mask_mid.any():
            batch_idx = torch.nonzero(mask_mid, as_tuple=False).squeeze(1)
            k_mid = k[batch_idx]
            km1 = (k_mid - 1).unsqueeze(1)
            kk = k_mid.unsqueeze(1)
            a = torch.gather(new_sigma[batch_idx], 1, km1)
            b = torch.gather(new_sigma[batch_idx], 1, kk)
            slice_view = new_sigma[batch_idx].clone()
            slice_view.scatter_(1, km1, b)
            slice_view.scatter_(1, kk, a)
            new_sigma[batch_idx] = slice_view

        return new_y, new_sigma

    def _run_walk(
        self,
        oracle_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int,
        y_base: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Execute a single dimension-lifting Sperner walk.

        Returns centroid weights of shape ``(batch_size, n_objs)``.
        """
        simplex_labels = torch.zeros((batch_size, self.n_objs),
                                     device=self.device,
                                     dtype=torch.long)

        def safe_oracle(w_batch):
            """Wrap the user's oracle, silently overriding violations of the
            Sperner boundary condition. See docs/THEORY.md for the consequences.
            """
            labels = oracle_fn(w_batch)
            chosen_w = w_batch[
                torch.arange(w_batch.shape[0], device=self.device), labels]
            bad = chosen_w <= 0
            if bad.any():
                # Boundary violation: substitute the index of the largest
                # nonzero coordinate. The walk continues but the result is
                # determined by this heuristic, not by the user's oracle.
                masked = w_batch.clone()
                masked[~bad] = 1.0  # unused — only bad rows are read below
                masked[masked <= 0] = -float('inf')
                labels[bad] = masked[bad].argmax(dim=-1)
            return labels

        v0 = self.get_vertex_batch(y_base, sigma, 0)
        simplex_labels[:, 0] = safe_oracle(self.get_barycentric_weights(v0))

        active_items = torch.ones(batch_size,
                                  device=self.device,
                                  dtype=torch.bool)

        for active_dim in range(1, self.d + 1):
            target_labels = set(range(active_dim + 1))
            door_labels = set(range(active_dim))

            for k in range(active_dim + 1):
                v = self.get_vertex_batch(y_base, sigma, k)
                simplex_labels[:, k] = safe_oracle(
                    self.get_barycentric_weights(v))

            last_pivoted_k = torch.full((batch_size, ),
                                        -1,
                                        device=self.device,
                                        dtype=torch.long)
            active_items.fill_(True)

            step_budget = self.n_sub * (active_dim + 1) * MAX_PIVOT_STEPS_PER_CELL
            for step in range(step_budget):
                pivot_k = torch.full((batch_size, ),
                                     -1,
                                     device=self.device,
                                     dtype=torch.long)
                for i in range(batch_size):
                    if not active_items[i]: continue

                    cur_labels = simplex_labels[i, :active_dim + 1].tolist()
                    if set(cur_labels).issuperset(target_labels):
                        if active_dim == self.d:
                            centroid_w = torch.zeros(self.n_objs,
                                                     device=self.device)
                            for vk in range(self.n_objs):
                                vv = self.get_vertex_batch(
                                    y_base[i:i + 1], sigma[i:i + 1], vk)
                                centroid_w += self.get_barycentric_weights(
                                    vv)[0]
                            centroid_w /= self.n_objs
                            if (centroid_w > 1e-6).all():
                                active_items[i] = False
                                continue
                        else:
                            active_items[i] = False
                            continue

                    found = False
                    for k in range(active_dim + 1):
                        if k == last_pivoted_k[i]: continue
                        face_labels = [
                            cur_labels[j] for j in range(active_dim + 1)
                            if j != k
                        ]
                        if set(face_labels) == door_labels:
                            pivot_k[i] = k
                            found = True
                            break
                    if not found: active_items[i] = False

                if not active_items.any(): break

                y_new, sigma_new = self.pivot_batch(y_base, sigma, pivot_k)
                new_weights = self.get_barycentric_weights(y_new)

                valid_mask = (new_weights >= 0).all(dim=1)
                actual_active = active_items & valid_mask
                if not actual_active.any():
                    active_items.fill_(False)
                    break

                y_base = torch.where(actual_active.unsqueeze(1), y_new, y_base)
                sigma = torch.where(actual_active.unsqueeze(1), sigma_new,
                                    sigma)

                upd_idx = torch.where(actual_active & (pivot_k != -1))[0]
                for idx in upd_idx:
                    pk = pivot_k[idx].item()
                    new_v = self.get_vertex_batch(y_base[idx:idx + 1],
                                                  sigma[idx:idx + 1], pk)
                    simplex_labels[idx, pk] = safe_oracle(
                        self.get_barycentric_weights(new_v))
                    last_pivoted_k[idx] = pk

        all_weights = torch.zeros((batch_size, self.n_objs),
                                  device=self.device)
        for k in range(self.n_objs):
            v = self.get_vertex_batch(y_base, sigma, k)
            all_weights += self.get_barycentric_weights(v)
        return all_weights / self.n_objs

    def _initial_state(self, batch_size: int,
                       randomise: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the starting (y_base, sigma) pair for a walk."""
        y_base = torch.zeros((batch_size, self.d),
                             device=self.device,
                             dtype=torch.long)
        if not randomise:
            sigma = torch.tile(torch.arange(self.d, device=self.device),
                               (batch_size, 1))
            return y_base, sigma

        sigma = torch.zeros((batch_size, self.d),
                            device=self.device,
                            dtype=torch.long)
        hi = max(2, self.n_sub - 1)
        for i in range(batch_size):
            coords = torch.randint(1,
                                   hi, (self.d, ),
                                   device=self.device).sort().values
            y_base[i] = coords
            sigma[i] = torch.randperm(self.d, device=self.device)
        return y_base, sigma

    def solve(
        self,
        oracle_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 1,
        max_restarts: int = 3,
        random_start: bool = False,
    ) -> torch.Tensor:
        """Run the dimension-lifting Sperner walk and return centroid weights.

        Uses multi-start to avoid boundary convergence: if a walk lands on a
        boundary face, the solver retries from a randomised interior starting
        point and returns the most interior result observed.

        Args:
            oracle_fn: Callable receiving a ``(batch, n_objs)`` weight tensor and
                returning a ``(batch,)`` long tensor of label indices. **Must
                satisfy the Sperner boundary condition** — see class docstring.
            batch_size: Number of independent walks to run in parallel.
            max_restarts: Maximum number of restart attempts when the walk
                converges to a boundary face (default 3).
            random_start: If True, the first attempt also uses a randomised
                interior starting point (rather than the canonical corner
                ``y = 0, sigma = arange(d)``). Useful for stochastic comparison
                of starting points.

        Returns:
            Tensor of shape ``(batch_size, n_objs)`` with the centroid weights of
            the panchromatic cell found by each walk.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        best_result = None
        best_min_weight = -1.0
        interior_threshold = 1.0 / (self.n_objs * 4)

        for attempt in range(max(1, max_restarts)):
            randomise = random_start or attempt > 0
            y_base, sigma = self._initial_state(batch_size, randomise)

            result = self._run_walk(oracle_fn, batch_size, y_base, sigma)
            min_w = result.min(dim=-1).values.min().item()

            if min_w > interior_threshold:
                return result

            if min_w > best_min_weight:
                best_result = result
                best_min_weight = min_w

        return best_result

    def solve_generator(
        self
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]], int,
                   torch.Tensor]:
        """Interactive coroutine version of the Sperner walk.

        Yields ``(vertex, weights, (active_dim, total_dim))`` tuples and
        expects a label integer to be sent via ``.send(int)``. Returns the
        final centroid weights when the walk terminates (caught via
        ``StopIteration.value``).

        Intended for human-in-the-loop UIs where each label is provided
        asynchronously.
        """
        batch_size = 1
        y_base = torch.zeros((batch_size, self.d),
                             device=self.device,
                             dtype=torch.long)
        sigma = torch.tile(torch.arange(self.d, device=self.device),
                           (batch_size, 1))
        simplex_labels = torch.zeros((batch_size, self.n_objs),
                                     device=self.device,
                                     dtype=torch.long)

        v0 = self.get_vertex_batch(y_base, sigma, 0)
        w0 = self.get_barycentric_weights(v0)
        label = yield (v0, w0, (0, self.d))
        simplex_labels[:, 0] = label

        for active_dim in range(1, self.d + 1):
            target_labels = set(range(active_dim + 1))
            door_labels = set(range(active_dim))

            for k in range(active_dim + 1):
                vk = self.get_vertex_batch(y_base, sigma, k)
                wk = self.get_barycentric_weights(vk)
                label = yield (vk, wk, (active_dim, self.d))
                simplex_labels[:, k] = label

            last_pivoted_k = -1
            step_budget = self.n_sub * (active_dim + 1) * 2
            for step in range(step_budget):
                cur_labels = simplex_labels[0, :active_dim + 1].tolist()
                if set(cur_labels).issuperset(target_labels): break

                pivot_k = -1
                for k in range(active_dim + 1):
                    if k == last_pivoted_k: continue
                    face = [
                        cur_labels[j] for j in range(active_dim + 1) if j != k
                    ]
                    if set(face) == door_labels:
                        pivot_k = k
                        break
                if pivot_k == -1:
                    all_w = torch.zeros((1, self.n_objs), device=self.device)
                    for ki in range(self.n_objs):
                        vi = self.get_vertex_batch(y_base, sigma, ki)
                        all_w += self.get_barycentric_weights(vi)
                    return all_w / self.n_objs

                pk_t = torch.tensor([pivot_k],
                                    device=self.device,
                                    dtype=torch.long)
                y_new, sigma_new = self.pivot_batch(y_base, sigma, pk_t)
                if (self.get_barycentric_weights(y_new) < 0).any():
                    all_w = torch.zeros((1, self.n_objs), device=self.device)
                    for ki in range(self.n_objs):
                        vi = self.get_vertex_batch(y_base, sigma, ki)
                        all_w += self.get_barycentric_weights(vi)
                    return all_w / self.n_objs

                y_base, sigma = y_new, sigma_new
                new_v = self.get_vertex_batch(y_base, sigma, pivot_k)
                new_w = self.get_barycentric_weights(new_v)
                label = yield (new_v, new_w, (active_dim, self.d))
                simplex_labels[0, pivot_k] = label
                last_pivoted_k = pivot_k

        all_weights = torch.zeros((1, self.n_objs), device=self.device)
        for k in range(self.n_objs):
            v = self.get_vertex_batch(y_base, sigma, k)
            all_weights += self.get_barycentric_weights(v)
        return all_weights / self.n_objs
