# ga.py
import numpy as np
import random
import copy

class WarehouseLayoutGA:
    def __init__(self, num_items=100, grid_size=(10, 10), population_size=50):
        """
        Initializes the Genetic Algorithm for Macro-Layout.
        :param num_items: Total number of unique items to store.
        :param grid_size: Tuple representing (rows, cols) of the warehouse grid.
        :param population_size: Number of candidate layouts in a generation.
        """
        self.num_items = num_items
        self.grid_size = grid_size
        self.population_size = population_size
        
        # Grid positions representing valid shelf locations
        # For simplicity, we assume every cell in the 10x10 grid is a valid shelf.
        self.shelf_locations = [(r, c) for r in range(grid_size[0]) for c in range(grid_size[1])]
        if len(self.shelf_locations) < num_items:
            raise ValueError("Grid is too small for the number of items.")
        self.shelf_locations = self.shelf_locations[:num_items]

    # ==========================================
    # 1. Representation & Initialization
    # ==========================================
    def generate_individual(self):
        """
        Generates a single feasible layout (chromosome).
        A layout is represented as a permutation of item IDs mapped to shelf locations.
        """
        # items are assumed to be 0 to num_items-1 or item IDs.
        layout = list(range(self.num_items))
        random.shuffle(layout)
        return layout

    def initialize_population(self):
        """Creates the initial population of warehouse layouts."""
        return [self.generate_individual() for _ in range(self.population_size)]

    # ==========================================
    # 2. Parent Selection
    # ==========================================
    def tournament_selection(self, population, fitnesses, k=3):
        """Selects a parent using k-way Tournament Selection."""
        selected_indices = random.sample(range(len(population)), k)
        best_index = min(selected_indices, key=lambda i: fitnesses[i])
        return copy.deepcopy(population[best_index])

    def roulette_wheel_selection(self, population, fitnesses):
        """Selects a parent using Fitness Proportionate (Roulette Wheel) Selection."""
        # For minimization, lower fitness is better. We invert the fitness.
        max_fit = max(fitnesses)
        inverted_fitness = [max_fit - f + 1e-6 for f in fitnesses] # +1e-6 to avoid 0
        total_fit = sum(inverted_fitness)
        probabilities = [f / total_fit for f in inverted_fitness]
        
        selected_index = np.random.choice(len(population), p=probabilities)
        return copy.deepcopy(population[selected_index])

    # ==========================================
    # 3. Crossover Operators
    # ==========================================
    def order_crossover_ox1(self, parent1, parent2):
        """Order Crossover (OX1)"""
        size = len(parent1)
        p1, p2 = sorted(random.sample(range(size), 2))
        
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy the slice
        child1[p1:p2] = parent1[p1:p2]
        child2[p1:p2] = parent2[p1:p2]
        
        # Fill the rest for child1 from parent2
        p2_idx = p2
        c1_idx = p2
        while -1 in child1:
            if parent2[p2_idx % size] not in child1:
                child1[c1_idx % size] = parent2[p2_idx % size]
                c1_idx += 1
            p2_idx += 1
            
        # Fill the rest for child2 from parent1
        p1_idx = p2
        c2_idx = p2
        while -1 in child2:
            if parent1[p1_idx % size] not in child2:
                child2[c2_idx % size] = parent1[p1_idx % size]
                c2_idx += 1
            p1_idx += 1
            
        return child1, child2

    def partially_mapped_crossover_pmx(self, parent1, parent2):
        """Partially Mapped Crossover (PMX)"""
        # Simplified PMX logic for permutations
        size = len(parent1)
        p1, p2 = sorted(random.sample(range(size), 2))
        
        def pmx_child(p_main, p_sub):
            child = [-1] * size
            child[p1:p2] = p_main[p1:p2]
            
            mapping = {p_main[i]: p_sub[i] for i in range(p1, p2)}
            
            for i in range(size):
                if p1 <= i < p2:
                    continue
                val = p_sub[i]
                while val in mapping:
                    val = mapping[val]
                child[i] = val
            return child
            
        return pmx_child(parent1, parent2), pmx_child(parent2, parent1)

    # ==========================================
    # 4. Mutation Operators
    # ==========================================
    def swap_mutation(self, individual, mutation_rate=0.05):
        """Swaps the locations of two items."""
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def inversion_mutation(self, individual, mutation_rate=0.05):
        """Reverses the order of a continuous section of the layout."""
        if random.random() < mutation_rate:
            p1, p2 = sorted(random.sample(range(len(individual)), 2))
            individual[p1:p2] = reversed(individual[p1:p2])
        return individual

    # ==========================================
    # 5. Survivor Selection & Diversity
    # ==========================================
    def elitism_replacement(self, population, fitnesses, new_population, new_fitnesses, keep_n=2):
        """Retains the best N individuals from the previous generation."""
        combined_pop = population + new_population
        combined_fit = fitnesses + new_fitnesses
        
        # Sort combined population by fitness
        sorted_indices = np.argsort(combined_fit)
        
        # Keep the top individuals
        next_gen = [combined_pop[i] for i in sorted_indices[:self.population_size]]
        next_fit = [combined_fit[i] for i in sorted_indices[:self.population_size]]
        
        return next_gen, next_fit
