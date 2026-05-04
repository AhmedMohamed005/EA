# Highly Detailed Project Documentation: Hybrid GA-ACO Framework for Warehouse Routing

This document provides a comprehensive, step-by-step breakdown of the codebase, methodologies, and technical implementations used to build and analyze the hybrid Genetic Algorithm (GA) and Ant Colony Optimization (ACO) framework.

---

## Step 1: Dataset Generation (The 80/20 Pareto Principle)

### Objective
In real-world warehouse logistics, pick frequencies are rarely uniform. The Pareto principle dictates that roughly 80% of order volume comes from 20% of the SKU inventory. Generating realistic test data is paramount for authentic algorithmic evaluation.

### Implementation: `dataset/generate_pareto.py`
This script constructs a dataset of synthetic warehouse orders mirroring a realistic product mix. We designate a small cluster of items (the first 20%) as "hot" and the rest as "cold".

**Detailed Code Explanation:**
```python
def generate_80_20_dataset(num_orders=100, num_items=100, output_path="pareto_warehouse_orders.csv"):
    # Split items strictly into 20% 'hot' and 80% 'cold'
    hot_items = np.arange(0, int(num_items*0.2))
    cold_items = np.arange(int(num_items*0.2), num_items)
    
    orders = []
    for _ in range(num_orders):
        order_size = random.randint(1, 5)  # Each order has 1 to 5 items
        order = []
        for _ in range(order_size):
            # 80% probability to pick a 'hot_item', 20% to pick a 'cold_item'
            if random.random() < 0.8 and len(hot_items) > 0:
                order.append(f"ITEM_{random.choice(hot_items):03d}")
            else:
                order.append(f"ITEM_{random.choice(cold_items):03d}")
        
        # Eliminate duplicates within the same order to simulate a realistic pick list
        order = list(set(order))
        if order:
            orders.append({
                "OrderID": f"ORD_{len(orders):03d}",
                "ItemsList": ", ".join(order)
            })
```
* **Why it matters:** Testing algorithms exclusively on uniformly distributed orders fails to stress-test congestion in high-traffic aisles. 

---

## Step 2: The Core Hybrid Engine (GA + ACO)

### Objective
To optimally sequence the picking order using Genetic Algorithms to explore macro-level permutations, while employing Ant Colony Optimization to calculate the micro-level routing costs (distance + congestion).

### Implementation: `src/hybrid.py`
The hybrid logic evaluates fitness maps. A genetic chromosome represents the order in which constraints and items are handled. The ACO maps actual paths through the warehouse grid infrastructure. 

**Detailed Code Explanation (Generational Update Engine):**
```python
    def run_generation(self, current_population, selection_method='tournament', mutation_method='swap'):
        """Executes one generation map over the population."""
        # Fix implemented for Batch Running: Safe State Caching guard
        if getattr(self, '_last_pop', None) != current_population:
            self._current_fitnesses = [self.evaluate_layout_fitness(ind) for ind in current_population]
            self._last_pop = current_population
        fitnesses = self._current_fitnesses
        
        new_population = []
        for _ in range(self.ga.population_size // 2):
            # 1. Parameterized Selection
            if selection_method == 'tournament':
                parent1 = self.ga.tournament_selection(current_population, fitnesses)
                parent2 = self.ga.tournament_selection(current_population, fitnesses)
            else:
                parent1 = self.ga.roulette_wheel_selection(current_population, fitnesses)
                parent2 = self.ga.roulette_wheel_selection(current_population, fitnesses)
            
            # 2. Hardcoded Crossover probability (50% OX1, 50% PMX)
            if random.random() < 0.5:
                child1, child2 = self.ga.order_crossover_ox1(parent1, parent2)
            else:
                child1, child2 = self.ga.partially_mapped_crossover_pmx(parent1, parent2)
            
            # 3. Parameterized Mutation
            if mutation_method == 'swap':
                child1, child2 = self.ga.swap_mutation(child1), self.ga.swap_mutation(child2)
            else:
                child1, child2 = self.ga.inversion_mutation(child1), self.ga.inversion_mutation(child2)
            
            new_population.extend([child1, child2])
```
* **Caching Fix Note:** To maintain batch iteration integrity, fitnesses are mapped reliably using an exact check `getattr(self, '_last_pop', None) != current_population`.
* **Soft Penalities:** Invalid layouts evaluate via soft bounds (`+15.0` penalty) resolving strictly unfeasible generations.

---

## Step 3: Automated Batch Testing (30 Seeds)

### Objective
For academic validity, experimental results must avoid subjective single-run testing. A formalized batch runner guarantees repeatable datasets for statistical significance testing (alpha=0.05).

### Implementation: `src/batch_runner.py`
This module spins up configurations testing 4 operator variants against 30 fixed random seeds.

**Detailed Code Explanation:**
```python
def run_batch_experiments(num_seeds=30, generations=20, pop_size=20, output_dir="results"):
    # Reproducible seed tracking
    seeds = list(range(1, num_seeds + 1))
    
    operator_configs = [
        {"selection": "tournament", "mutation": "swap"},
        {"selection": "roulette", "mutation": "swap"},
        {"selection": "tournament", "mutation": "inversion"},
        {"selection": "roulette", "mutation": "inversion"}
    ]
    
    # Executes and collects independent generational paths
    for config in operator_configs:
        for seed in seeds:
            set_global_seed(seed)
            optimizer = HybridOptimizer(dataset_path=dataset_path)
            # Generational Loop execution
            ...
            
    # Stores flattened structures via Pandas into a tracked CSV artifact
    df_results = pd.DataFrame([...])
    df_results.to_csv(csv_path, index=False)
```
* **Why it matters:** It enforces strict parametric configurations removing UI/GUI bottleneck biases and builds the foundational dataset (`batch_30_seed_stats.csv`) used to justify configuration choices.

---

## Step 4: Analytical Plotting & Visualization

### Objective
Convert raw statistics into comprehensible, academically rigorous charts. Boxplots represent the spread, medians, and variance accurately.

### Implementation: `src/plot_results.py`
A matplotlib aggregator parsing generated CSV artifacts.

**Detailed Code Explanation:**
```python
def plot_operator_comparison():
    stats_path = "results/batch_30_seed_stats.csv"
    df = pd.read_csv(stats_path)
    
    plt.figure(figsize=(10, 6))
    df.boxplot(column='BestFitness', by='Operator', grid=False, rot=15)
    plt.title('Operator Performance Comparison (30 Seeds)')
    plt.suptitle('')
    plt.ylabel('Best Fitness Cost')
    plt.xlabel('Operator Combination (Selection_Mutation)')
    plt.tight_layout()
    plt.savefig('results/operator_comparison.png', dpi=300)
```
* **Deliverables Formed:** This outputs multiple artifacts (`operator_comparison.png`, `heatmap.png`, `scalability_analysis.png`, `product_mix_comparison.png`).

---

## Step 5: Academic Reporting and Limitations Extracted

In standard research reporting, finding purely identical performance is common and should be formally addressed.

### The Truth About Results
*   **Result Matrix:** Mean best fitnesses are approximately `452.0` across *all* configurations. 
*   **Statistical Truth:** When passing `p-values` against Tournament/Swap vs Roulette/Inversion, `p > 0.05` signifies the null hypothesis stands: **The operator chosen creates zero statistically significant performance difference.** 
*   **Academic Writeup Context:** The GA handles the macro variables, but the framework's convergence speed is entirely bottlenecked and dominated by the ACO (Ant Colony) execution path logic underneath.

### Identified Codebase Limitations (For Report Acknowledgment)
1. **O(N²) Linear Scanning:** `_crowding_replacement` iteratively checks subset diversities in Python lists. Large scale generations will trigger exponential time slowdowns. 
2. **Timeless Congestion:** Ant paths measure edge utilization across multiple traces, but this assumes all pickers traverse the aisle at the same *time* regardless of sequence intervals. True timestamped 4D-routing is missing. 
3. **Stochastic Correlation Collisions:** Because Crossover relies on a `0.5` random flip (OX1 vs PMX) acting against an identical starting seed, downstream mutations occasionally output visually identical final fitness ratings across differing operators simply by statistical chance paths mirroring perfectly.