import os
import sys
import torch
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from equilib.ndim_topo_align import NDimTopoAlignSolver

def simulate_llm_eval(weights: np.ndarray):
    """
    Mock LLM evaluator. In a real scenario, this would:
    1. Merge model adapters using the weights.
    2. Run a benchmark (e.g., MMLU, GSM8K, SafetyBench).
    3. Return the scores.
    """
    # Simple model of trade-offs:
    # w0 (Helpfulness), w1 (Safety), w2 (Verbosity)
    w_h, w_s, w_v = weights
    
    # 1. Helpfulness is best when mostly w0, but hurt by w1 (too much safety filter).
    score_h = 0.8 * w_h + 0.2 * w_v - 0.3 * w_s + 0.1
    
    # 2. Safety is best when mostly w1.
    score_s = 0.9 * w_s - 0.1 * w_h + 0.1
    
    # 3. Verbosity increases with w0 and w2, decreases with w1.
    score_v = 0.6 * w_v + 0.4 * w_h - 0.2 * w_s
    
    return np.clip([score_h, score_s, score_v], 0, 1)

def main():
    print("--- Equilib Quickstart: 3-Objective Alignment ---")
    
    # Target Metrics: We want Helpfulness=0.7, Safety=0.8, Verbosity=0.4
    targets = np.array([0.7, 0.8, 0.4])
    print(f"Target Performance: Helpfulness={targets[0]}, Safety={targets[1]}, Verbosity={targets[2]}")

    # Initialize the N-Dimensional Solver (n_objs=3)
    # subdivision=50 defines the precision of the search grid.
    solver = NDimTopoAlignSolver(n_objs=3, subdivision=50)

    # Define the "Judge" (Oracle Function)
    # This function tells Equilib which objective is currently MOST UNSATISFIED.
    def rlhf_judge(w_batch: torch.Tensor):
        # w_batch has shape (batch_size, 3)
        batch_size = w_batch.shape[0]
        labels = []
        
        for i in range(batch_size):
            # Convert torch weights to numpy for our simulator
            w = w_batch[i].cpu().numpy()
            
            # 1. Run our evaluation
            current_metrics = simulate_llm_eval(w)
            
            # 2. Calculate "Gaps" (how far we are from targets)
            gaps = targets - current_metrics
            
            # 3. Choose the objective that needs more weight (largest gap)
            label = np.argmax(gaps)
            print(f"  Step: Weights={np.round(w, 2)} -> Gaps={np.round(gaps, 2)} -> Label={label}")
            labels.append(label)
            
        return torch.tensor(labels, device=w_batch.device, dtype=torch.long)

    # Run the topological solve
    print("\nRunning topological walk (Sperner Walk)...")
    result_batch = solver.solve(oracle_fn=rlhf_judge, batch_size=1)
    
    # Extract the optimal weights
    optimal_weights = result_batch[0].cpu().numpy()
    
    # Final Validation
    final_metrics = simulate_llm_eval(optimal_weights)
    
    print("\n" + "="*40)
    print(" OPTIMIZATION COMPLETED")
    print("="*40)
    print(f"Optimal Weights:  H={optimal_weights[0]:.3f}, S={optimal_weights[1]:.3f}, V={optimal_weights[2]:.3f}")
    print(f"Final Metrics:    H={final_metrics[0]:.3f}, S={final_metrics[1]:.3f}, V={final_metrics[2]:.3f}")
    print(f"Target Metrics:   H={targets[0]:.3f}, S={targets[1]:.3f}, V={targets[2]:.3f}")
    print("="*40)

if __name__ == "__main__":
    main()
