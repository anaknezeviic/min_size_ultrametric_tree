import numpy as np
from itertools import product
import os

class Tree:
    def __init__(self):
        self.tree = {}
        self.parents = {}

    def add_edge(self, parent, child, weight):
        if child in self.tree and any(p == parent for p, w in self.tree[child]):
            return

        if parent not in self.tree:
            self.tree[parent] = []
        self.tree[parent].append((child, weight))
        if child not in self.parents:
            self.parents[child] = []
        self.parents[child].append((parent, weight))

    def calculate_total_weight(self):
        total_weight = sum(weight for parents in self.parents.values() for parent, weight in parents)
        return total_weight

    def find_path_weight(self, start, end):
        visited = set()
        return self._dfs(start, end, visited, 0)
    
    def _dfs(self, current, target, visited, current_weight):
        if current == target:
            return current_weight
        visited.add(current)
        for neighbor, weight in self.tree.get(current, []):
            if neighbor not in visited:
                result = self._dfs(neighbor, target, visited, current_weight + weight)
                if result is not None:
                    return result
        
        parent_info = self.parents.get(current, [])
        if parent_info and parent_info[0][0] not in visited:
            return self._dfs(parent_info[0][0], target, visited, current_weight + parent_info[0][1])
        
        return None

    def generate_all_combinations(self):
        parent_values = [(p, c, w) for c, parents in self.parents.items() for p, w in parents]
        
        all_combinations = set(product(parent_values, repeat=len(self.parents)))

        valid_subtrees = []
        for combo in all_combinations:
            sub_tree = Tree()
            valid = True
            visited = set()

            for p, c, w in combo:
                if c in visited or p == c or sub_tree.has_cycle(p, c):
                    valid = False
                    break
                visited.add(c)
                sub_tree.add_edge(p, c, w)

            if valid:
                valid_subtrees.append(sub_tree)

        return valid_subtrees

    def has_cycle(self, parent, child):
        if parent == child:
            return True
        
        visited = set()
        return self._detect_cycle(child, parent, visited)
    
    def _detect_cycle(self, node, target, visited):
        if node in visited:
            return False
        visited.add(node)
        
        if node in self.tree:
            for child, _ in self.tree[node]:
                if child == target or self._detect_cycle(child, target, visited):
                    return True
        return False

    def __hash__(self):
        edges = sorted([(p, c, w) for p, children in self.tree.items() for c, w in children])
        return hash(tuple(edges))
    
    def __eq__(self, other):
        if isinstance(other, Tree):
            return hash(self) == hash(other)
        return False
    
    def __repr__(self):
        edges = []
        for parent, children in self.tree.items():
            for child, weight in children:
                edges.append(f"{parent} --({weight})--> {child}")
        return "\n".join(edges)


def is_ultrametric(tree, M):
    root_candidates = set(tree.tree.keys()) - set(tree.parents.keys())

    if not root_candidates:
        return False

    leaves = set(tree.parents.keys()) - set(tree.tree.keys())

    for root in root_candidates:
        distances = set()
        is_valid_ultrametric = True

        for leaf in leaves:
            distance = tree.find_path_weight(root, leaf)
            distances.add(distance)

            for other_leaf in leaves:
                if leaf != other_leaf:
                    dij = tree.find_path_weight(leaf, other_leaf)
                    if dij < M[leaf][other_leaf]:
                        is_valid_ultrametric = False
                        break
            if not is_valid_ultrametric:
                break

        if len(distances) == 1 and is_valid_ultrametric:
            return True

    return False


def generate_unique_ultrametric_subtrees(best_tree, M):
    subtrees = best_tree.generate_all_combinations()
    unique_ultrametric_subtrees = set(subtree for subtree in subtrees if is_ultrametric(subtree, M))
    return unique_ultrametric_subtrees

def brute_force_for_minimum_ultrametric_tree(M):
    n = len(M)
   
    initial_tree = Tree()
    for i in range(n):
        for j in range(n):
            if i != j:
                if (i not in initial_tree.parents or initial_tree.parents[i][0] != j) and \
                   (j not in initial_tree.parents or initial_tree.parents[j][0] != i):
                    initial_tree.add_edge(i, j, M[i, j])
 
    ultrametric_subtrees = generate_unique_ultrametric_subtrees(initial_tree, M)
    min_weight = float('inf')
    minimum_ultrametric_tree = None

    for subtree in ultrametric_subtrees:
        current_weight = subtree.calculate_total_weight()
        if current_weight < min_weight:
            min_weight = current_weight
            minimum_ultrametric_tree = subtree


    return minimum_ultrametric_tree, min_weight

def load_distance_matrix(file_path):
    return np.loadtxt(file_path, delimiter=' ')

def load_all_distance_matrices(directory_path):
    matrices = {}
    for file_name in os.listdir(directory_path):
        if file_name.startswith('matrix') and file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            matrices[file_name] = load_distance_matrix(file_path)
    return matrices

directory_path = 'tests/'
distance_matrices = load_all_distance_matrices(directory_path)

for file_name, distance_matrix in distance_matrices.items():
    print(f"Processing matrix from file: {file_name}")
    optimal_tree, min_weight = brute_force_for_minimum_ultrametric_tree(distance_matrix)
    if optimal_tree != None:
        print(optimal_tree)
        print("Weight of the optimal tree:", min_weight)
    else:
        print("No ultrametric tree found.")

