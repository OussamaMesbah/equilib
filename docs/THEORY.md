# Topological Fixed-Point Theory for Model Alignment

## Overview

The Equilib library (Topo-Align) treats the problem of multi-objective model alignment not as a scalar optimization task, but as a search for a fixed-point equilibrium within a simplicial complex. This approach is grounded in combinatorial topology and specifically utilizes Sperner's Lemma and the Brouwer Fixed-Point Theorem.

## The Simplicial Complex

Alignment between $n$ objectives is represented on an $(n-1)$-dimensional simplex. Each vertex of the simplex represents a 100% weighting of a single objective (e.g., Safety, Coding, or Creativity). Any point within the simplex represents a convex combination of these objectives, where the sum of weights equals 1.

## Sperner's Lemma and Labeling

The core of the algorithm is the Sperner Labeling Rule. For a triangulation of the simplex, each vertex $v$ is assigned a label $L(v) \in \{0, 1, \dots, n-1\}$.

A labeling is a "Sperner Labeling" if:
1. Each main vertex $V_i$ of the simplex has label $i$.
2. Any vertex $v$ located on a face of the simplex only has a label corresponding to one of the vertices defining that face.

In the context of model alignment, the label assigned to a weight configuration $w$ is the index of the objective that is "most dissatisfied" or "most neglected" at that point.

## The Sperner Walk

Sperner's Lemma guarantees that in any Sperner-labeled triangulation, there exists at least one "panchromatic" simplex—a small cell containing all $n$ labels. This cell represents the equilibrium where no single objective is more dissatisfied than the others.

The algorithm finds this cell by performing a "walk" from the boundary of the simplex. By moving through adjacent simplices that share a "door" (a face labeled with a specific subset of required labels), the walk is mathematically guaranteed to converge to the equilibrium point.

## Advantages Over Gradient-Based Optimization

1. **Derivative-Free**: No need for a differentiable loss function.
2. **Qualitative Input**: The solver only requires a directional "label" (which objective to improve) rather than a precise numerical score.
3. **Global Convergence**: Unlike gradient descent, which can get stuck in local minima, the topological walk is guaranteed to find a balanced solution within the defined bounds.
