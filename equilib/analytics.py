"""Alignment diagnostics and walk analytics."""

from typing import List, Sequence, Union

import numpy as np


def calculate_frustration_score(
    path_vertices: Sequence[Union[List[float], np.ndarray]], ) -> float:
    """Measure topological frustration of a Sperner walk path.

    The frustration score is the ratio of total path length to net displacement.

    * ~1.0 — direct convergence, minimal conflict.
    * 1.5–3.0 — moderate trade-off complexity.
    * >3.0 — high frustration; objectives are strongly conflicting.
    * 999.0 — loop detected (zero net displacement).

    Args:
        path_vertices: Sequence of coordinate arrays visited by the solver.

    Returns:
        Frustration score (float).  Returns 1.0 for paths shorter than 2 steps.
    """
    if not path_vertices or len(path_vertices) < 2:
        return 1.0

    path = np.array(path_vertices)

    # 1. Calculate total distance walked (sum of Euclidean steps)
    diffs = path[1:] - path[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    total_dist = np.sum(distances)

    # 2. Calculate displacement (Start to Finish)
    start = path[0]
    end = path[-1]
    displacement = np.linalg.norm(end - start)

    # Avoid division by zero
    if displacement < 1e-9:
        return 999.0  # Loop detected or returned to start

    # Ratio: How much did we wander?
    frustration_index = total_dist / displacement

    return float(frustration_index)
