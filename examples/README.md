# Examples

This directory contains standalone scripts demonstrating the capabilities of the Equilib library.

- `rlhf_steering_demo.py`: Simulates a Multi-Objective RLHF environment (Helpfulness, Safety, Verbosity) and uses the topological solver to find optimal reward mixing weights.
- `generate_sperner_dataset.py`: The utility used to create the `sperner-bench` dataset, demonstrating how to generate high-dimensional simplex triangulations.

## How to run
From the repository root:
```bash
python examples/rlhf_steering_demo.py
```
