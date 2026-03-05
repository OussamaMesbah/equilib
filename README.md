# Equilib: PyTorch-Native Topological Manifold Alignment

**Equilib** is a high-performance engine for finding the "Sweet Spot" in multi-objective AI systems—such as balancing Code Quality vs. Safety in Model Merging, or Precision vs. Recall in RAG.

It uses **Topological Alignment** (based on Sperner's Lemma) to navigate complex trade-offs without needing gradients, making it ideal for metrics like "vibe checks," LLM-as-a-Judge, or discrete benchmarks.

## 🚀 Why Equilib?

### The Problem: The "Alignment Tax"
In modern LLMs, optimizing for one task (e.g., Creative Writing) often degrades another (e.g., Factuality). Classical optimization fails here because:
*   **Gradient Descent** requires differentiable functions (most LLM benchmarks are not).
*   **Grid Search** is too slow (exponentially grows with the number of objectives).
*   **Bayesian Optimization** becomes computationally expensive as you add more objectives.

### The Solution: Topological Search
Equilib treats the balance of $N$ objectives as a coordinate in a simplex. Instead of "guessing" weights, it performs a **Sperner Walk**—a combinatorial search that is guaranteed to find the equilibrium point where all objectives are balanced.

| Feature | Grid Search | Bayesian Opt | **Equilib** |
| :--- | :--- | :--- | :--- |
| **Speed** | Slow ($O(X^N)$) | Medium | **Fast ($O(N)$)** |
| **Gradients** | Not needed | Not needed | **Not needed** |
| **Scalability** | Poor | Average | **High (10+ objectives)** |
| **Guarantee** | No | Probabilistic | **Mathematical (Sperner's Lemma)** |

---

## 📦 Installation

```bash
pip install .
# Or for development
pip install -e ".[dev]"
```

## 🛠 Usage Workflows

Equilib supports three primary workflows: **Automated**, **Manual (Human-in-the-Loop)**, and **CLI-based**.

### 1. Automated Alignment (Code)
Use this when you have programmatic evaluators (e.g., a benchmark suite or an LLM-as-a-Judge).

```python
from equilib import NDimTopoAlignSolver

# 1. Define your "Judge" (Oracle)
# It returns the INDEX of the objective that needs more weight.
def my_judge(weights):
    # weights = [0.2, 0.5, 0.3] for 3 objectives
    scores = evaluate_model(weights) # e.g., {"safety": 0.8, "coding": 0.4, "chat": 0.6}
    # Coding (index 1) has the lowest score
    return 1 

# 2. Solve for the Equilibrium
solver = NDimTopoAlignSolver(n_objs=3)
optimal_weights = solver.solve(oracle_fn=my_judge)
print(f"Optimal Balance: {optimal_weights}")
```

### 2. Manual "Vibe Checks" (Human-in-the-Loop)
When metrics are purely qualitative (e.g., "Is this response too robotic?"), use the built-in Streamlit UI to steer the model manually.

#### **How it works:**
1.  **Generation**: Equilib proposes a set of weights and generates a response from your model.
2.  **Verdict**: You (the human) look at the response and click a button for the objective that is **currently failing** (e.g., "Needs more Creativity").
3.  **Pivot**: Equilib uses your feedback to mathematically pivot the model's latent manifold toward the next test point.
4.  **Convergence**: In a few steps, you reach the "Goldilocks Zone" where all objectives are satisfied.

#### **How to run it:**
```bash
# 1. Ensure you have a local LLM server running (e.g., LM Studio, Ollama)
# 2. Run the UI
streamlit run app.py
```
*Also available as a demo on **Hugging Face Spaces**.*

### 3. Model Merging (CLI)
Balance LoRA adapters or model weights directly from the terminal.

```bash
python tools/topo_merge.py --base meta-llama/Llama-3-8B --adapters coding-lora,safety-lora --precision 100
```

---

## 🧠 Core Features

*   **Topological MoE Routing**: Bypasses unstable softmax gating by finding the Nash Equilibrium of expert contributions per token.
*   **Simplicial LoRA Fusion**: Resolves the "Alignment Tax" by calculating Pareto-optimal weights for merging $N$ specialized adapters.
*   **PyTorch-Native**: Fully tensorized for CUDA-accelerated batch processing.

## Architecture

Equilib operates on the principle that the optimal capability mix of an LLM lies at a fixed point within a simplicial complex (**Implicit Freudenthal Triangulation**). It navigates the manifold based on discrete "most dissatisfied objective" labels, ensuring convergence even in noisy, non-convex landscapes.

## Citation
```bibtex
@software{mesbah2026equilib,
  author = {Mesbah, Oussama},
  title = {Equilib: High-Performance Topological Alignment},
  year = {2026},
  url = {https://github.com/omesbah/topo-align}
}
```
