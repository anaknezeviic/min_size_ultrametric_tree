import numpy as np
import os
from functools import lru_cache
import time

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
        self.parents[child] = (parent, weight)

    def calculate_total_weight(self):
        return sum(weight for _, weight in self.parents.values())

    @lru_cache(None)
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
        
        parent_info = self.parents.get(current)
        if parent_info and parent_info[0] not in visited:
            return self._dfs(parent_info[0], target, visited, current_weight + parent_info[1])
        
        return None

    def __repr__(self):
        edges = []
        for parent, children in self.tree.items():
            for child, weight in children:
                edges.append(f"{parent} --({weight})--> {child}")
        return "\n".join(edges)

def maxmin_permutation(M):
    n = len(M)
    a1, a2 = np.unravel_index(np.argmax(M, axis=None), M.shape)
    permutation = [a1, a2]
    
    min_dists = np.min(M, axis=1)
    
    while len(permutation) < n:
        remaining_elements = [k for k in range(n) if k not in permutation]
        next_element = max(remaining_elements, key=lambda k: min(M[k, ai] for ai in permutation))
        permutation.append(next_element)
    
    return permutation


def upgmm(M):
    n = len(M)
    clusters = {i: [i] for i in range(n)}
    tree = Tree()
    current_node = n 
    heights = {i: 0 for i in range(n)}

    while len(clusters) > 1:
        min_dist = float('inf')
        best_pair = None

        for i in clusters:
            for j in clusters:
                if i != j:
                    dist = max([M[x, y] for x in clusters[i] for y in clusters[j]])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (i, j)

        if best_pair is not None:
            i, j = best_pair
            new_cluster = clusters[i] + clusters[j]
            new_height = min_dist / 2
            tree.add_edge(current_node, i, new_height - heights[i])
            tree.add_edge(current_node, j, new_height - heights[j])
            heights[current_node] = new_height
            del clusters[i]
            del clusters[j]
            clusters[current_node] = new_cluster
            current_node += 1

    for i in range(n):
        for j in range(i + 1, n):
            path_weight = tree.find_path_weight(i, j)
    
    return tree

def compute_lower_bound(M, inserted_leaves):
    bound = 0
    for i in range(1, len(inserted_leaves)):
        min_weight = min(M[inserted_leaves[j], inserted_leaves[i]] for j in range(i))
        bound += min_weight // 2 
    return bound

def is_ultrametric(tree, M):
    root_candidates = set(tree.tree.keys()) - set(tree.parents.keys())
    if not root_candidates:
        return False
    
    root = root_candidates.pop()
    leaves = set(tree.parents.keys()) - set(tree.tree.keys())
    distances = set()

    for leaf in leaves:
        distance = tree.find_path_weight(root, leaf)
        distances.add(distance)

    for i in leaves:
        for j in leaves:
            if i != j:
                d_ij = tree.find_path_weight(i, j)
                if d_ij < M[i, j]:
                    return False

    return len(distances) == 1

def branch_and_bound(M):
    n = len(M)
    permutation = maxmin_permutation(M)
    initial_tree = Tree()
    for i in range(n):
        for j in range(n):
            if i != j:
                if (i not in initial_tree.parents or initial_tree.parents[i][0] != j) and \
                   (j not in initial_tree.parents or initial_tree.parents[j][0] != i):
                    initial_tree.add_edge(i, j, M[i, j])

    root_tree = Tree()
    if permutation[0] in initial_tree.tree:
        for child, weight in initial_tree.tree[permutation[0]]:
            if child == permutation[1]:
                root_tree.add_edge(permutation[0], permutation[1], weight)
    
    root_bound = compute_lower_bound(M, permutation[:2])
    UB_tree = upgmm(M)
    UB = UB_tree.calculate_total_weight()
    
    pq = [(root_bound, root_tree, 2)]
    best_tree = UB_tree
    
    while pq:
        pq.sort(key=lambda x: x[0])
        current_bound, current_tree, depth = pq.pop(0)
        
        if current_bound >= UB:
            continue
        
        if depth == n:
            if is_ultrametric(current_tree, M):
                tree_weight = current_tree.calculate_total_weight()
                UB = tree_weight
                best_tree = current_tree
            continue
        
        for i in range(depth, n):
            new_leaf = permutation[i]
            for existing_leaf in permutation[:depth]:
                if (existing_leaf in initial_tree.tree and 
                    any(child == new_leaf for child, weight in initial_tree.tree[existing_leaf])):
                    
                    new_tree = insert_leaf(current_tree, existing_leaf, new_leaf, M[existing_leaf, new_leaf])
                    
                    if new_tree is not None:
                        new_bound = compute_lower_bound(M, permutation[:depth+1])
                        if new_bound < UB:
                            pq.append((new_bound, new_tree, depth + 1))
    
    return best_tree, UB

def insert_leaf(tree, parent, new_leaf, weight):
    new_tree = Tree()
    new_tree.tree = {k: v[:] for k, v in tree.tree.items()}
    new_tree.parents = tree.parents.copy()
    
    if new_leaf in new_tree.parents:
        return None
    
    new_tree.add_edge(parent, new_leaf, weight)
    
    root_candidates = set(new_tree.tree.keys()) - set(new_tree.parents.keys())
    if len(root_candidates) > 1:
        return None
    
    return new_tree

def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

def load_distance_matrix(file_path):
    return np.loadtxt(file_path, delimiter=' ')

def load_all_distance_matrices(directory_path):
    matrices = {}
    for file_name in os.listdir(directory_path):
        if file_name.startswith('matrix') or file_name.startswith('bbu'):
            file_path = os.path.join(directory_path, file_name)
            matrices[file_name] = load_distance_matrix(file_path)
    return matrices

directory_path = 'tests/' 
distance_matrices = load_all_distance_matrices(directory_path)

for file_name, distance_matrix in distance_matrices.items():
    print(f"Processing matrix from file: {file_name}")
    (optimal_tree, time_needed) = measure_execution_time(branch_and_bound, distance_matrix)
    print(optimal_tree[0])
    print(f"Weight of the optimal tree: {optimal_tree[1]}")
    print(f"Time needed: {time_needed}")