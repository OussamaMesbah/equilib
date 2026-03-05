import logging
from typing import List, Optional

import numpy as np

from .solver import EquilibSolver

logger = logging.getLogger(__name__)


class AdaptiveEquilibSolver(EquilibSolver):
    """Iterative-refinement ("zoom") solver for high-precision 3-objective alignment.

    Repeatedly solves on coarser grids and re-bases the search simplex onto the
    panchromatic triangle found in the previous iteration, achieving exponential
    precision improvement with linear cost.

    Args:
        subdivision: Base grid resolution per zoom level.
        max_depth: Maximum number of zoom iterations.
        precision: Target diameter for the final triangle (in weight space).
    """

    def __init__(
        self,
        subdivision: int = 10,
        max_depth: int = 5,
        precision: float = 1e-6,
    ) -> None:
        super().__init__(subdivision)
        self.max_depth = max_depth
        self.precision = precision
        self.basis = np.eye(3)

    def weights_from_coords(self, x, y):
        """
        Maps local grid coordinates (x, y) to Global Weights via the current Basis.
        """
        # 1. Local Barycentric Coordinates (u, v, w)
        u = x / self.n
        v = y / self.n
        w = (self.n - x - y) / self.n

        local_weights = np.array([u, v, w])

        # 2. Map to Global Simplex via Matrix Multiplication
        return local_weights @ self.basis

    def solve_adaptive(self):
        """
        Runs the iterative 'Zoom' process.
        """
        logger.info(
            f"Starting Adaptive Sperner (Depth {self.max_depth}, Grid {self.n})..."
        )

        final_tri = None
        global_tri_weights = []

        for depth in range(1, self.max_depth + 1):
            logger.info(f"DEPTH {depth}: Zooming into sub-simplex...")
            # Run the standard walk on the current basis
            result_tri_coords, path = self.walk()

            if not result_tri_coords:
                logger.error("FAIL: Walk failed at this depth.")
                break

            # Extract the vertices of the result triangle in GLOBAL weights
            # The result_tri_coords are integer tuples [(x1,y1), (x2,y2), (x3,y3)]
            global_tri_weights = []
            vertex_labels = []

            for pt in result_tri_coords:
                g_w = self.weights_from_coords(*pt)
                label = self.oracle_label(*pt)
                global_tri_weights.append(g_w)
                vertex_labels.append(label)

            # Visualization of current precision
            d01 = np.linalg.norm(global_tri_weights[0] - global_tri_weights[1])
            d12 = np.linalg.norm(global_tri_weights[1] - global_tri_weights[2])
            d20 = np.linalg.norm(global_tri_weights[2] - global_tri_weights[0])
            max_diam = max(d01, d12, d20)

            # Calculate centroid
            centroid = sum(global_tri_weights) / 3
            logger.info(
                f"RESULT Depth {depth}: Centroid {np.round(centroid, 5)} | Precision (Diam): {max_diam:.6f}"
            )

            if max_diam < self.precision:
                logger.info(
                    f"DONE: Precision target {self.precision} reached.")
                break

            # PREPARE NEXT DEPTH: "Zoom" into this triangle

            # Check if we have a panchromatic triangle (labels {0, 1, 2})
            if set(vertex_labels) != {0, 1, 2}:
                logger.warning(
                    f"WARN: Triangle at depth {depth} is not panchromatic: {vertex_labels}. Zooming might fail."
                )
                break

            new_basis = np.zeros((3, 3))

            for w, l in zip(global_tri_weights, vertex_labels):
                new_basis[l] = w

            self.basis = new_basis
            self.vertices = {}

            final_tri = global_tri_weights

        if final_tri:
            logger.info(
                f"COMPLETE: Final High-Precision Equilibrium: {np.round(sum(final_tri)/3, 7)}"
            )
        return final_tri


if __name__ == "__main__":
    # Run Adaptive Solver
    # Start with a coarse grid (n=10) but zoom in 10 times.
    solver = AdaptiveEquilibSolver(subdivision=10,
                                   max_depth=10,
                                   precision=1e-7)
    solver.solve_adaptive()
