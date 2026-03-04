# System Architecture

## Core Modules

### 1. NDimTopoAlignSolver (`equilib.ndim_topo_align`)
This is the central mathematical engine of the library. It implements the N-dimensional Sperner walk using an implicit Freudenthal triangulation.
- **Implicit Grid**: Calculates neighbors "on-the-fly" using algebraic pivoting, avoiding memory issues associated with large explicit grids ($O(N)$ memory scaling).
- **Dimension Lifting**: Solves high-dimensional problems by recursively solving smaller sub-simplex boundaries.

### 2. SpernerTrainer (`equilib.sperner_trainer`)
A high-level utility for integrating the topological solver with PEFT (Parameter-Efficient Fine-Tuning) libraries.
- **LoRA Support**: Automatically merges multiple LoRA adapters using the weights discovered by the solver.
- **Objective Wrapper**: Converts standard loss functions or human preference judges into labels required by the solver.

### 3. NDimSurrogateTopoAlignSolver (`equilib.surrogate_topo_align`)
Accelerates alignment in high-dimensional or expensive objective spaces.
- **Active Learning**: Uses K-Nearest Neighbors (KNN) or Random Forests to learn a surrogate model of the alignment landscape.
- **Oracle Optimization**: Reduces the number of calls to the "expensive" real-world objective function (e.g., human-in-the-loop or slow LLM judge).

### 4. Human-in-the-Loop UI (`equilib.human_ui`)
A Streamlit-based graphical interface for qualitative "vibe-check" alignment.
- **Real-Time Pivot**: Allows a human user to navigate the weight space by simply identifying the "least satisfied" objective in a set of model outputs.

## Data Flow

1. **Initialization**: Define objectives (e.g., Safety, Coding) and subdivision resolution.
2. **Oracle Request**: The solver requests a label for a specific weight configuration.
3. **Evaluation**:
    - *Automated:* Run inference on a merged model and calculate which objective metric is lowest.
    - *Human:* Present model outputs to a human judge who selects the failing objective.
4. **Pivoting**: The solver executes a topological pivot to move to an adjacent simplex.
5. **Convergence**: Once all labels $\{0, 1, \dots, n-1\}$ are found in a single simplex, the solver returns the centroid of that simplex as the optimal weighting.
