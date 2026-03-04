# Equilib: PyTorch-Native Topological Manifold Alignment

Equilib is a high-performance engine for resolving multi-objective equilibria in foundation models. It utilizes an **Implicit Freudenthal Triangulation** to navigate (N-1)-dimensional simplicial complexes without the memory overhead of explicit mesh generation.

The primary application is the resolution of **Destructive Interference** during continuous Mixture-of-Experts (MoE) routing and latent LoRA fusion in non-differentiable or discrete feedback environments.

## Technical Specifications

*   **Engine**: PyTorch-Native (Tensorized)
*   **Complexity**: $O(N)$ memory per pivot, where $N$ is the number of objectives.
*   **Parallelism**: Support for CUDA-accelerated batch processing (batch solve).
*   **Convergence**: Guaranteed convergence to a panchromatic simplex via Sperner's Lemma.
*   **Numerical Stability**: Integrated Pareto-tracking fallback for noisy or non-convex manifold landscapes.

## Core Features

### 1. Topological MoE Routing
Bypasses unstable softmax gating by finding the absolute Nash Equilibrium of expert contributions per token. 

```python
from equilib import NDimTopoAlignSolver

# Initialize batch solver on GPU
solver = NDimTopoAlignSolver(n_objs=8, device="cuda")

# Vectorized solve for 1024 tokens in parallel
routing_weights = solver.solve(oracle_fn=starvation_judge, batch_size=1024)
```

### 2. Simplicial LoRA Fusion
Resolves the "Alignment Tax" by calculating the Pareto-optimal weights for merging $N$ specialized adapters.

```python
from equilib.industrial import AutoModelMerger

merger = AutoModelMerger(base_model="llama-3", adapters=["code", "safety"])
weights = merger.find_optimal_mix(evaluators=[humaneval, safetybench])
```

## CLI Usage

The `topo-merge` utility provides a zero-code entry point for automated adapter balancing.

```bash
python tools/topo_merge.py --base meta-llama/Llama-3-8B --adapters coding-lora,safety-lora --precision 100
```

## Architecture

Equilib operates on the principle that the optimal capability mix of an LLM lies at a fixed point within a simplicial complex. Traditional gradient descent often fails in these spaces due to the non-differentiability of qualitative metrics (e.g., "vibe checks"). Equilib instead performs a combinatorial search (Sperner Walk) that navigates the manifold based on discrete "most dissatisfied objective" labels.

### Pivot Logic
The solver implements algebraic pivoting rules on Kuhn triangulation indices, ensuring that each step remains within the simplex bounds while moving toward the panchromatic cell.

## Citation

```bibtex
@software{mesbah2026equilib,
  author = {Mesbah, Oussama},
  title = {Equilib: High-Performance Topological Alignment},
  year = {2026},
  url = {https://github.com/omesbah/topo-align}
}
```
