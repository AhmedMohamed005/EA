# hybrid.py
import random
import numpy as np
import pandas as pd
from .ga import WarehouseLayoutGA
from .aco import WarehouseACO

def set_global_seed(seed):
    """
    Centralized random seed management for reproducibility (Requirement for the 30-Run Rule).
    Aligns with standard academic experimental protocols.
    """
    random.seed(seed)
    np.random.seed(seed)

class HybridOptimizer:
    def __init__(self, dataset_path, grid_size=(10, 10), hazardous_items=None, safe_zone_rows=None):
        """
        Integrates the GA (Layout) with ACO (Routing) to form the Hybrid solver.
        :param dataset_path: Path to the orders CSV.
        :param grid_size: Dimensions of the warehouse.
        :param hazardous_items: List of item IDs considered hazardous.
        :param safe_zone_rows: Allowed row indices for hazardous items.
        """
        self.grid_size = grid_size
        self.num_items = grid_size[0] * grid_size[1]
        
        # Load dataset
        self.df_orders = pd.read_csv(dataset_path)
        self.all_orders = []
        
        # Map Item strings (e.g. ITEM_001) to integer indices (0 to 99)
        self.item_to_idx = {}
        self.idx_to_item = {}
        self._parse_orders()
        
        # Initialize sub-algorithms
        self.ga = WarehouseLayoutGA(num_items=self.num_items, grid_size=grid_size)
        
        # Constraints parameters
        self.hazardous_items = hazardous_items or []
        self.safe_zone_rows = safe_zone_rows or [grid_size[0] - 1] # By default, back wall
        self.packing_zone_rows = [grid_size[0] - 2] # Explicit packing zone separate from hazardous
        self.penalty_weight = 1000.0  # Heavy penalty for constraint violations

    def _parse_orders(self):
        """Iterates over the dataset and maps items to integer indices for GA."""
        item_counter = 0
        for current_idx, row in self.df_orders.iterrows():
            items = row['ItemsList'].split(', ')
            order_indices = []
            for item in items:
                if item not in self.item_to_idx:
                    self.item_to_idx[item] = item_counter
                    self.idx_to_item[item_counter] = item
                    item_counter += 1
                order_indices.append(self.item_to_idx[item])
            self.all_orders.append(order_indices)

    def calculate_penalty(self, layout):
        """
        Penalty Function (Constraints Handling).
        Checks if hazardous materials are placed outside the safe designated zones.
        Also penalizes placing regular items in the packing zone.
        """
        penalty = 0.0
        for i, item_idx in enumerate(layout):
            r, c = self.ga.shelf_locations[i]
            item_str = self.idx_to_item.get(item_idx, "")
            
            # Hazardous constraint
            if item_str in self.hazardous_items:
                if r not in self.safe_zone_rows:
                    penalty += self.penalty_weight
            
            # Packing zone constraint: items placed here incur a soft operational cost, 
            # rather than a hard infeasibility penalty, because all 100 cells must be used in a 10x10 grid.
            if r in self.packing_zone_rows:
                penalty += 15.0  # Soft penalty (operational cost) instead of hard penalty
                
        return penalty

    def evaluate_layout_fitness(self, layout, sample_size=10):
        """
        Fitness Function connecting GA and ACO.
        1. Maps layout to 2D grid coordinates.
        2. Applies Penalty Functions for constraints.
        3. Uses ACO to calculate traveling distance and time for a batch of orders.
        4. Calculates congestion penalties.
        """
        item_locations = {}
        for slot_idx, item_idx in enumerate(layout):
            item_locations[item_idx] = self.ga.shelf_locations[slot_idx]
            
        aco = WarehouseACO(num_nodes=self.num_items + 1)
        
        sorted_locs = [item_locations[i] for i in range(self.num_items)]
        aco.initialize_graph(sorted_locs, start_point=(0, 0))
        
        sample_orders = random.sample(self.all_orders, min(sample_size, len(self.all_orders)))
        total_time = 0.0
        edge_usage = {}
        
        for order in sample_orders:
            required_nodes = [idx + 1 for idx in order]
            route, distance = aco.simulate_ant_routing(required_nodes)
            
            # Picking time: 2 seconds per item
            picking_time = len(order) * 2.0
            
            # Assume ant speed is 1 meter per second, so distance = time
            route_time = distance + picking_time
            total_time += route_time
            
            # Track congestion
            for i in range(len(route) - 1):
                u, v = route[i], route[i+1]
                edge = tuple(sorted((u, v)))
                edge_usage[edge] = edge_usage.get(edge, 0) + 1
                
        # Congestion penalty: penalize edges used too frequently
        congestion_penalty = sum((count - 1) * 5.0 for count in edge_usage.values() if count > 1)
        
        constraint_penalty = self.calculate_penalty(layout)
        
        return total_time + congestion_penalty + constraint_penalty

    def _crowding_replacement(self, current_pop, current_fit, new_pop, new_fit):
        """Simple diversity preservation using distance checking."""
        combined_pop = current_pop + new_pop
        combined_fit = current_fit + new_fit
        
        unique_pop = []
        unique_fit = []
        
        for p, f in zip(combined_pop, combined_fit):
            if p not in unique_pop:
                unique_pop.append(p)
                unique_fit.append(f)
                
        sorted_indices = np.argsort(unique_fit)
        
        next_gen = [unique_pop[i] for i in sorted_indices[:self.ga.population_size]]
        next_fit = [unique_fit[i] for i in sorted_indices[:self.ga.population_size]]
        
        # If we removed too many duplicates, fill with random to maintain size
        while len(next_gen) < self.ga.population_size:
            ind = self.ga.generate_individual()
            next_gen.append(ind)
            next_fit.append(self.evaluate_layout_fitness(ind))
            
        return next_gen, next_fit

    def run_generation(self, current_population, selection_method='tournament', mutation_method='swap'):
        """Executes one generation map over the population."""
        # Always evaluate if coming from a fresh set (removes stale cache guard safely)
        if getattr(self, '_last_pop', None) != current_population:
            self._current_fitnesses = [self.evaluate_layout_fitness(ind) for ind in current_population]
            self._last_pop = current_population
        fitnesses = self._current_fitnesses
        
        new_population = []
        for _ in range(self.ga.population_size // 2):
            if selection_method == 'tournament':
                parent1 = self.ga.tournament_selection(current_population, fitnesses)
                parent2 = self.ga.tournament_selection(current_population, fitnesses)
            else: # roulette
                parent1 = self.ga.roulette_wheel_selection(current_population, fitnesses)
                parent2 = self.ga.roulette_wheel_selection(current_population, fitnesses)
            
            if random.random() < 0.5:
                child1, child2 = self.ga.order_crossover_ox1(parent1, parent2)
            else:
                child1, child2 = self.ga.partially_mapped_crossover_pmx(parent1, parent2)
            
            if mutation_method == 'swap':
                child1 = self.ga.swap_mutation(child1)
                child2 = self.ga.swap_mutation(child2)
            else: # inversion
                child1 = self.ga.inversion_mutation(child1)
                child2 = self.ga.inversion_mutation(child2)
            
            new_population.extend([child1, child2])
            
        new_fitnesses = [self.evaluate_layout_fitness(ind) for ind in new_population]
        
        # Apply crowd-based elitism for diversity and survivor selection
        next_population, next_fitnesses = self._crowding_replacement(
            current_population, fitnesses, new_population, new_fitnesses
        )
        self._current_fitnesses = next_fitnesses
        self._last_pop = next_population
        
        return next_population, next_fitnesses

    def get_sample_route(self, layout):
        """
        Purely for GUI Visualization. 
        Returns an ACO route for a single sample order.
        """
        item_locations = {}
        for slot_idx, item_idx in enumerate(layout):
            item_locations[item_idx] = self.ga.shelf_locations[slot_idx]
            
        aco = WarehouseACO(num_nodes=self.num_items + 1)
        sorted_locs = [item_locations[i] for i in range(self.num_items)]
        aco.initialize_graph(sorted_locs, start_point=(0, 0))
        
        if not self.all_orders:
            return [], sorted_locs
            
        order = self.all_orders[0]  # Just visualize the first order tracking
        required_nodes = [idx + 1 for idx in order]
        best_route, _ = aco.simulate_ant_routing(required_nodes)
        
        return best_route, sorted_locs
