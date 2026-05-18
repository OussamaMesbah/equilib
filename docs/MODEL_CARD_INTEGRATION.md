# Model Card Integration Template

If you have used Sperner to find a balanced mixing-weight vector for your
model, copy the following section into your Hugging Face model card. **Be
honest about what Sperner does**: it returns a single centroid of a
panchromatic cell on the simplex of mixing weights, given a deterministic
oracle that points at the "most underserved" objective. It is not an
alignment method by itself, and it is not a Nash equilibrium solver.

---

## Adapter mixing

This model uses adapter mixing weights computed with
[Sperner](https://github.com/OussamaMesbah/sperner), a PyTorch implementation
of the Kuhn-Freudenthal Sperner walk for finding panchromatic-cell centroids
on the simplex.

### Objectives

We treated the following capabilities as the `n` objectives of the simplex:

- [Objective 1: e.g., Safety]
- [Objective 2: e.g., Python Coding]
- [Objective 3: e.g., Logical Reasoning]

### Mixing weights

The centroid of the panchromatic cell discovered by the walk was:

- **Adapter A:** [Weight A]
- **Adapter B:** [Weight B]
- **Adapter C:** [Weight C]

These weights are blended into the base model parameters at inference time.

### Oracle and methodology

The walk used a [Human / deterministic-metric / cached-benchmark] oracle that,
for any candidate mix `w`, returns the index of the objective with the
largest deficit relative to its target score. The oracle satisfies the
**Sperner boundary condition** (label `i` is never returned at points where
`w_i = 0`).

Grid resolution: `subdivision = [N]`. With the library's loop bound, the
walk evaluated at most `~ N · d²` candidate mixes, where `d = n - 1`.

### Caveats

- This procedure finds **one balanced point**, not the Pareto frontier of
  capability trade-offs. Other balanced points may exist.
- The result depends on the oracle's choice of "most underserved" objective.
  Changing how that scalar is computed will change the centroid.
- For noisy oracles (e.g., human raters), repeat runs will produce different
  centroids. Report the variation across seeds.

### Reproducibility

```python
import numpy as np
from sperner import solve_equilibrium

def my_oracle(w):
    scores = my_evaluation_function(w)
    scores[w <= 0] = -np.inf  # Sperner boundary
    return int(np.argmax(my_targets - scores))

weights = solve_equilibrium(n_objs=3, subdivision=50, oracle=my_oracle)
```
