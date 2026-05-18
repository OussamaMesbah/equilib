# Sperner

**A PyTorch implementation of Sperner / Kuhn-Freudenthal walks for finding panchromatic-cell centroids on labeled simplices.**

[![Tests](https://github.com/OussamaMesbah/sperner/actions/workflows/test.yml/badge.svg)](https://github.com/OussamaMesbah/sperner/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/sperner.svg)](https://pypi.org/project/sperner/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this library does

Given a **cheap, deterministic oracle** that, for any weight vector `w` on the
`(n-1)`-simplex, returns the index of one objective `i ∈ {0, …, n-1}` — Sperner finds
a small simplicial cell whose vertices cover **all** `n` labels (a *panchromatic cell*),
and returns its centroid.

By Sperner's Lemma (1928), such a cell exists for any labeling that satisfies the
**Sperner boundary condition**:

> On the face `w_i = 0`, the oracle must not return label `i`.

When the labeling is induced by a continuous map `f: Δ → Δ`, the centroid of the
panchromatic cell approximates a **Brouwer fixed point** of that map (via the
Knaster-Kuratowski-Mazurkiewicz theorem). The approximation error shrinks with grid
resolution.

That is what this library computes. **Nothing more.**

### What this library is *not*

- **It is not a Nash-equilibrium solver.** Nash equilibria are fixed points of
  best-response correspondences in strategic games; this library does not model
  any game. Earlier versions of this README used the term "Nash equilibrium" loosely
  — that was wrong, and has been removed.
- **It is not an LLM alignment method.** Real alignment (RLHF, DPO, Constitutional
  AI, etc.) shapes model parameters from gradient or preference signals. This
  library only computes a single point on the simplex of *mixing weights*, given
  a hand-written oracle. See [§ Honest limits](#honest-limits) below.
- **It is not "O(N)".** Sperner walks are PPAD-complete in general (Papadimitriou
  1994). The implementation's loop bound is `O(n_sub · d²)` pivots per walk for `d = n-1`.
  See [§ Complexity](#complexity).

---

## When this library is genuinely useful

Use Sperner when **all of the following** hold:

1. You have a multi-objective problem and want **one balanced operating point**, not a Pareto frontier.
2. Your oracle is **cheap and deterministic** (e.g., a closed-form metric, a small benchmark, a cached evaluation).
3. You can phrase your oracle as *"at weights w, which objective is most underserved?"* — and the answer satisfies the Sperner boundary condition.
4. The number of objectives is small to moderate (≤ ~10 in practice — beyond that, the grid pivots get expensive).

Concretely: balancing precision/recall/latency in a deterministic classifier;
finding a mixing point between several closed-form scoring rules; teaching
combinatorial topology with an interactive Streamlit walk.

For **noisy** oracles or **expensive** evaluators (LLM benchmarks, human
judgments), see the surrogate solver — but read its caveats first, because the
silent boundary override means a misbehaving real oracle won't crash the walk,
it will just produce a meaningless centroid.

---

## Installation

```bash
pip install sperner

# Latest from source
pip install git+https://github.com/OussamaMesbah/sperner.git

# Local editable install
pip install -e .

# With LoRA/PEFT support (experimental, see caveats below)
pip install "sperner[peft]"

# With Streamlit human-in-the-loop UI
pip install "sperner[ui]"

# Everything
pip install "sperner[all]"
```

## Quick start

```python
import numpy as np
import torch
from sperner import solve_equilibrium

torch.manual_seed(0)

# Argmax-gap labeling — a textbook Sperner labeling whose continuous
# triple-point sits at `target`.
target = np.array([0.4, 0.4, 0.2])

def oracle(w: np.ndarray) -> int:
    # Sperner boundary condition: do not return label i if w[i] == 0.
    gaps = target - w
    gaps[w <= 0] = -np.inf
    return int(np.argmax(gaps))

weights = solve_equilibrium(n_objs=3, subdivision=50, oracle=oracle)
# `weights` is the centroid of a panchromatic cell found by the walk.
# It is *near* but not necessarily equal to `target` — the walk finds
# *some* panchromatic cell, not specifically the one containing the
# triple-point of the labeling. See docs/THEORY.md for the caveats.
print(weights)
```

### Programmatic batch API

```python
import torch
from sperner import NDimEquilibSolver

solver = NDimEquilibSolver(n_objs=4, subdivision=30)

def judge(weights_batch: torch.Tensor) -> torch.Tensor:
    """Return label index per row in the batch.

    Must satisfy: weights_batch[i, label[i]] > 0 for all i (Sperner boundary).
    """
    labels = []
    for w in weights_batch:
        scores = my_deterministic_metric(w.numpy())
        # argmin score = "most underserved"; mask out zero-weight objectives.
        scores_masked = np.where(w.numpy() > 0, scores, np.inf)
        labels.append(int(np.argmin(scores_masked)))
    return torch.tensor(labels)

result = solver.solve(oracle_fn=judge, batch_size=1)
print(result[0])
```

### Human-in-the-loop (Streamlit UI)

For interactive exploration — the solver proposes weights, your local LLM
generates a response at those weights, and you pick which objective is
currently weakest. This is an **educational tool**, not an alignment pipeline.

```bash
streamlit run app.py
```

The UI supports 2–10 configurable objectives and works with any OpenAI-compatible
API (LM Studio, Ollama, vLLM).

### LoRA adapter merging (experimental)

```python
from sperner import SpernerTrainer

trainer = SpernerTrainer(
    base_model=my_peft_model,
    adapters=["safety-lora", "code-lora", "chat-lora"],
    objectives=[safety_score, code_score, chat_score],
    mock=False,  # See warnings below
)
optimal_mix = trainer.train(grid_size=30)
```

> **Caveat:** with `mock=False`, every pivot step calls every objective on the
> blended model. For 3 objectives on a 30-cell grid you can expect hundreds of
> full-model evaluations — usually slower than a coarse direct sweep, and a tiny
> fraction of what RLHF/DPO does in the same time. The trainer emits a
> `warnings.warn` when invoked in non-mock mode.

---

## Complexity

The `_run_walk` loop in [sperner/ndim_solver.py](sperner/ndim_solver.py) bounds
pivot steps by `n_sub · (active_dim + 1) · MAX_PIVOT_STEPS_PER_CELL` per
dimension-lifting phase, summed over `d = n-1` phases:

```
oracle calls ≲ MAX_PIVOT_STEPS_PER_CELL · n_sub · d · (d + 1) / 2
            ≈ O(n_sub · d²)
```

Concretely, for `n_objs = 10, subdivision = 50` the loop can take up to
**~9,000 pivots** in the worst case (in practice the walk terminates earlier
on well-behaved oracles). For the legacy 2D solver, the bound in
[sperner/solver.py](sperner/solver.py) is `O(n_sub²)`.

Sperner walks are **PPAD-complete** in general (Papadimitriou 1994) — there is no
known polynomial-time algorithm for finding panchromatic cells in arbitrary
labelings. The empirical numbers above are typical-case, not worst-case.

### Comparison to alternatives

|                                | Grid search (resolution X) | Sperner (this library) |
| :----------------------------- | :------------------------- | :--------------------- |
| Oracle calls (typical case)    | `O(X^N)`                   | `O(X · N²)` in `n_sub` and `d`   |
| Worst-case guarantee           | Exhaustive                 | PPAD-complete          |
| Needs gradients                | No                         | No                     |
| Returns                        | Full landscape             | One centroid           |
| Tolerates noisy oracle         | Naturally                  | No — Sperner condition required |

For **noisy oracles**, **Pareto fronts**, or **>10 objectives**, prefer
multi-objective Bayesian optimization (BoTorch's qNEHVI/ParEGO) or
evolutionary methods (NSGA-II via [pymoo](https://pymoo.org/)).

---

## Theory and citations

The algorithm is a textbook **Scarf/Kuhn fixed-point walk** on the
Kuhn-Freudenthal triangulation:

- **Sperner, E.** (1928). *Neuer Beweis für die Invarianz der Dimensionszahl und des Gebietes*. Abh. Math. Sem. Hamburg, 6: 265–272.
- **Scarf, H.** (1967). *The Approximation of Fixed Points of a Continuous Mapping*. SIAM J. Appl. Math., 15(5): 1328–1343.
- **Kuhn, H. W.** (1968). *Simplicial approximation of fixed points*. PNAS, 61(4): 1238–1242.
- **Papadimitriou, C. H.** (1994). *On the complexity of the parity argument and other inefficient proofs of existence*. JCSS, 48(3): 498–532. (PPAD-completeness.)
- **Freudenthal, H.** (1942). *Simpliziale Zerlegungen von beschränkter Flachheit*. Annals of Mathematics, 43(3): 580–582. (Triangulation used here.)

See [docs/THEORY.md](docs/THEORY.md) for a careful statement of what Sperner's
Lemma does and does not give you, including the relationship to KKM and Brouwer.

---

## Honest limits

The earlier marketing for this project significantly overstated the case.
The current README is written with full awareness that:

- The oracle contract — *"return the most underserved objective"* — is awkward
  for objectives that are **not on the same scale**. There is no canonical way
  to compare "safety = 0.7" with "helpfulness = 0.9" without per-objective
  normalization, and the library does not provide one.
- Real LLM judges are **stochastic**. Stochastic labels break the panchromatic
  guarantee — repeated calls at the same `w` can yield different labels,
  invalidating the walk's termination conditions.
- The library **silently overrides** oracle labels that violate the Sperner
  boundary condition (in `ndim_solver._run_walk`, `solver.oracle_label`, and
  `moe_router.forward_route`). If your oracle returns label `i` at a point
  where `w_i = 0`, the library quietly substitutes a different label. The
  walk then converges, but to a point determined by the override heuristic,
  not by your oracle. This is documented in
  [docs/THEORY.md](docs/THEORY.md#the-boundary-condition-and-silent-overrides)
  and the relevant docstrings.
- The MoE router in `sperner/moe_router.py` runs a full Sperner walk per
  routed input. It is a **research demo**, not a production routing layer —
  softmax routing is many orders of magnitude faster and the routing-collapse
  problem has much cheaper fixes (load-balancing loss, expert dropout).
  The class now emits a warning when called.

If any of these limits matter for your use case, this is not the right tool.

---

## Project structure

```
sperner/
  ndim_solver.py       # Core N-dimensional Sperner walk (PyTorch)
  solver.py            # Legacy 2D solver (3 objectives, configurable target)
  adaptive_solver.py   # Iterative zoom refinement (3-objective only)
  surrogate_solver.py  # KNN active-learning wrapper for expensive oracles
  sperner_trainer.py   # PEFT/LoRA adapter integration (experimental)
  moe_router.py        # Topological MoE routing (research demo)
  agentic_judge.py     # Synthetic-oracle helper for batch demos
  human_ui.py          # Streamlit UI for manual labeling
  analytics.py         # Walk-path frustration analysis
  plotting.py          # Simplex heatmap visualization (3D only)
  industrial.py        # Wrapper for adapter mixing by user-supplied evaluators
```

## Docs

- [API Reference](docs/API_REFERENCE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Theory: Sperner / KKM / Brouwer](docs/THEORY.md)
- [Model Card integration template](docs/MODEL_CARD_INTEGRATION.md)

## Citation

```bibtex
@software{mesbah2026sperner,
  author = {Mesbah, Oussama},
  title = {Sperner: PyTorch implementation of Kuhn-Freudenthal Sperner walks},
  year = {2026},
  url = {https://github.com/OussamaMesbah/sperner}
}
```

## Support

If you find this implementation useful for teaching, research demos, or
small deterministic balancing problems, you can support development at:

[<img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" >](https://www.buymeacoffee.com/omesbahf)

## License

MIT
