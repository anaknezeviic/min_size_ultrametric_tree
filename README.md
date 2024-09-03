# Minimum size ultrametric tree

## Problem definition
INSTANCE:  $n\times n$ matrix M of positive integers.

SOLUTION: An ultrametric tree, i.e., an edge-weighted tree T(V,E) with n leaves such that, for any pair of leaves i and j, $d_{ij}^T \geq M[i,j]$ where $d_{ij}^T$ denotes the sum of the weights in the path between i and j.

MEASURE: The size of the tree, i.e., $\sum_{e \in E} w(e)$ where w(e) denotes the weight of edge e. 

## Optimization Techniques
**1. Brute force:** Exhaustive search algorithm evaluating all possible solutions.

**2. Branch-and-bound:** Tree search algorithm who repeatedly searches the branch-and-bound tree for better solutions until an (nearly) optimal solution is found.

**3. Genetic algorithm:** The genetic algorithm is an optimization method that use selection, crossover and mutation to iteratively improve the population of trees. 
## Usage
1. Clone repository.
   
2. Install the required dependencies.
 
3. Run the desired optimization technique.

## Contributing
[Ana Knezevic](https://github.com/anaknezeviic)

[Pavle Ponjavic](https://github.com/pavlee12)
