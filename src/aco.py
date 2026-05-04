# aco.py
import numpy as np
import random

class WarehouseACO:
    def __init__(self, num_nodes, alpha=1.0, beta=2.0, evaporation_rate=0.1, q=100.0, num_ants=10):
        """
        Initializes the Ant Colony Optimization for Micro-Routing.
        :param num_nodes: Total number of locations (items + start/end points).
        :param alpha: Relative importance of pheromone (\tau).
        :param beta: Relative importance of heuristic information (\eta).
        :param evaporation_rate: Rate at which pheromone evaporates (\rho).
        :param q: Pheromone deposit factor.
        :param num_ants: Number of ants to simulate per route.
        """
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        self.q = q
        self.num_ants = num_ants
        
        # Pheromone matrix \tau
        self.pheromones = np.ones((num_nodes, num_nodes))
        # Heuristic visibility matrix \eta (1 / distance)
        self.visibility = np.zeros((num_nodes, num_nodes))
        # Distance matrix
        self.distances = np.zeros((num_nodes, num_nodes))

    def initialize_graph(self, item_locations, start_point=(0, 0)):
        """
        Builds the grid graph reflecting the GA layout output.
        Calculates distances and initial visibility.
        :param item_locations: List of (x, y) tuples representing item positions.
        :param start_point: (x, y) tuple for the shipping dock/start location.
        """
        # Node 0 is the start point, nodes 1 to N are items
        all_nodes = [start_point] + item_locations
        self.num_nodes = len(all_nodes)
        
        self.pheromones = np.ones((self.num_nodes, self.num_nodes))
        self.distances = np.zeros((self.num_nodes, self.num_nodes))
        self.visibility = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    # Manhattan distance is commonly used for warehouse grids
                    dist = abs(all_nodes[i][0] - all_nodes[j][0]) + abs(all_nodes[i][1] - all_nodes[j][1])
                    if dist == 0:
                        dist = 1e-4 # Avoid division by zero
                    self.distances[i][j] = dist
                    self.visibility[i][j] = 1.0 / dist

    def transition_probability(self, current_node, unvisited_nodes):
        """
        Calculates the probability of moving to unvisited nodes.
        :param current_node: The node the ant is currently on.
        :param unvisited_nodes: List of node indices yet to be visited.
        """
        probabilities = []
        denominator = 0.0
        
        for node in unvisited_nodes:
            tau = self.pheromones[current_node][node] ** self.alpha
            eta = self.visibility[current_node][node] ** self.beta
            score = tau * eta
            probabilities.append(score)
            denominator += score
            
        if denominator == 0:
            # Fallback to random if probabilities collapse
            return [1.0 / len(unvisited_nodes)] * len(unvisited_nodes)
            
        return [p / denominator for p in probabilities]

    def simulate_ant_routing(self, required_items):
        """
        Simulates ants picking a batch of orders.
        :param required_items: List of node indices (1 to N) the ant must visit.
        :return: Tuple of (best_route, best_distance)
        """
        best_route = []
        best_distance = float('inf')
        all_routes = []
        all_distances = []
        
        for ant in range(self.num_ants):
            current_node = 0  # Start at shipping dock
            unvisited = set(required_items)
            route = [current_node]
            route_distance = 0.0
            
            while unvisited:
                unvisited_list = list(unvisited)
                probs = self.transition_probability(current_node, unvisited_list)
                
                # Roulette wheel selection for next node
                next_node = np.random.choice(unvisited_list, p=probs)
                
                route_distance += self.distances[current_node][next_node]
                current_node = next_node
                
                route.append(current_node)
                unvisited.remove(current_node)
                
            # Return to start point
            route_distance += self.distances[current_node][0]
            route.append(0)
            
            all_routes.append(route)
            all_distances.append(route_distance)
            
            if route_distance < best_distance:
                best_distance = route_distance
                best_route = route
                
        self.update_pheromones(all_routes, all_distances)
        return best_route, best_distance

    def update_pheromones(self, routes, distances):
        """
        Evaporates old pheromones and deposits new pheromones based on route quality.
        """
        # 1. Evaporation
        self.pheromones = self.pheromones * (1.0 - self.rho)
        
        # 2. Deposition
        for route, dist in zip(routes, distances):
            deposit = self.q / dist
            for i in range(len(route) - 1):
                u, v = route[i], route[i+1]
                self.pheromones[u][v] += deposit
                self.pheromones[v][u] += deposit # symmetric
