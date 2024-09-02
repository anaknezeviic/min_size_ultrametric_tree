import random
import numpy as np
import os

class Tree:
    def __init__(self, n):
        self.n = n
        self.edges = {}
        self.fitness = None

    def add_edge(self, u, v, weight):
        if (v, u) in self.edges:
            return 
        self.edges[(u, v)] = weight

    def calculate_total_weight(self):
        return sum(self.edges.values())

    def is_ultrametric(self, matrix):
        n = len(matrix) 
        for (u, v), weight in self.edges.items():
            if u < n and v < n:
                if weight < matrix[u][v]:
                    return False
            else:
                return False
        return True

    def calculate_balance_penalty(self):
        leaf_distances = []

        def calculate_leaf_distance(node, current_distance, visited):
            if node in visited:
                return  
            visited.add(node)
            
            children = [child for parent, child in self.edges if parent == node]
            if not children: 
                leaf_distances.append(current_distance)
            else:
                for child in children:
                    weight = self.edges[(node, child)]
                    calculate_leaf_distance(child, current_distance + weight, visited)
        
        
        calculate_leaf_distance(0, 0, set())  
        
        if not leaf_distances:
            return float('inf')
        
        max_distance = max(leaf_distances)
        min_distance = min(leaf_distances)
        
        return (max_distance - min_distance) ** 2



    def calc_fitness(self, matrix):
        if not self.is_ultrametric(matrix):
            return float('-inf')  
        total_weight = self.calculate_total_weight()
        edge_count_penalty = abs(len(self.edges) - (self.n - 1)) * 10  
        balance_penalty = self.calculate_balance_penalty()  
        self.fitness = -total_weight - edge_count_penalty - balance_penalty

        return self.fitness

    def __repr__(self):
        return f"Tree with edges: {self.edges} and fitness: {self.fitness}"


class Individual:
    def __init__(self, matrix=None, tree=None):
        if tree is None:
            self.n = len(matrix)
            self.tree = self.build_tree_from_matrix(matrix)
        else:
            self.n = tree.n
            self.tree = tree

        self.fitness = self.tree.calc_fitness(matrix if matrix is not None else np.zeros((self.n, self.n)))

    def build_tree_from_matrix(self, matrix):
        tree = Tree(self.n)
        used_nodes = set()
        remaining_nodes = set(range(self.n))
        
        current_node = 0
        used_nodes.add(current_node)
        remaining_nodes.remove(current_node)
        
        while remaining_nodes:
            nearest_node = None
            nearest_distance = float('inf')
            for node in used_nodes:
                for other_node in remaining_nodes:
                    if matrix[node][other_node] < nearest_distance:
                        nearest_distance = matrix[node][other_node]
                        nearest_node = other_node
                        connecting_node = node

            tree.add_edge(connecting_node, nearest_node, nearest_distance)
            used_nodes.add(nearest_node)
            remaining_nodes.remove(nearest_node)
        
        return tree

def selection(population, tournament_size):
    chosen = random.sample(population, tournament_size)
    return max(chosen, key=lambda x: x.fitness)

def crossover(parent1, parent2):
    child_tree1 = Tree(parent1.n)
    child_tree2 = Tree(parent1.n)

    for edge in parent1.tree.edges:
        if edge in parent2.tree.edges:
            if random.random() < 0.5:
                child_tree1.add_edge(*edge, parent1.tree.edges[edge])
                child_tree2.add_edge(*edge, parent2.tree.edges[edge])
            else:
                child_tree1.add_edge(*edge, parent2.tree.edges[edge])
                child_tree2.add_edge(*edge, parent1.tree.edges[edge])
        else:
            child_tree1.add_edge(*edge, parent1.tree.edges[edge])
            child_tree2.add_edge(*edge, parent1.tree.edges[edge])

    for edge in parent2.tree.edges:
        if edge not in child_tree1.edges:
            child_tree1.add_edge(*edge, parent2.tree.edges[edge])
            child_tree2.add_edge(*edge, parent2.tree.edges[edge])

    return Individual(tree=child_tree1), Individual(tree=child_tree2)

def mutation(individual, matrix, mutation_prob):
    n = len(matrix)
    for edge in list(individual.tree.edges):
        if random.random() < mutation_prob:
            u, v = edge
            new_weight = random.randint(matrix[u][v], max(matrix.flatten()))
            individual.tree.edges[edge] = new_weight
        if random.random() < mutation_prob:
            u, v = random.sample(range(n), 2)
            if (u, v) not in individual.tree.edges and u != v:
                individual.tree.add_edge(u, v, random.randint(matrix[u][v], max(matrix.flatten())))

    individual.fitness = individual.tree.calc_fitness(matrix)


def load_distance_matrix(file_path):
    return np.loadtxt(file_path, delimiter=' ')

def load_all_distance_matrices(directory_path):
    matrices = {}
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            matrices[file_name] = load_distance_matrix(file_path)
    return matrices

def genetic_algorithm(matrix, population_size, num_generations, tournament_size , elitism_size, mutation_prob):
    population = [Individual(matrix) for _ in range(population_size)]
    best_individual = max(population, key=lambda x: x.fitness)

    for generation in range(num_generations):
        new_population = []

        population.sort(key=lambda x: x.fitness, reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1 = selection(population, tournament_size)
            parent2 = selection(population, tournament_size)
            child1, child2 = crossover(parent1, parent2)

            mutation(child1, matrix, mutation_prob)
            mutation(child2, matrix, mutation_prob)

            child1.fitness = child1.tree.calc_fitness(matrix)
            child2.fitness = child2.tree.calc_fitness(matrix)

            new_population.extend([child1, child2])

        population = new_population[:population_size]
        current_best = max(population, key=lambda x: x.fitness)

        if current_best.fitness > best_individual.fitness:
            best_individual = current_best


    return best_individual

directory_path = 'tests/'
distance_matrices = load_all_distance_matrices(directory_path)

for file_name, distance_matrix in distance_matrices.items():
    print(f"Processing matrix from file: {file_name}")
    best_individual = genetic_algorithm(
        matrix=distance_matrix,
        population_size=100,
        num_generations=15,
        tournament_size=7,
        elitism_size=10,
        mutation_prob=0.05
    )
    if best_individual.tree:
        print("The best tree:")
        print(best_individual.tree)
        print(f"Weight of the optimal tree: {abs(best_individual.fitness)}")
    else:
        print("No ultrametric tree found.")
