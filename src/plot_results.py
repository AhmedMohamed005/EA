# plot_results.py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from .hybrid import HybridOptimizer, set_global_seed

def plot_heatmap(optimizer, best_layout):
    """
    Plots a heatmap of the pheromone matrix overlaid on the warehouse layout to
    visualize congestion and visit frequencies.
    """
    item_locations = {}
    for slot_idx, item_idx in enumerate(best_layout):
        item_locations[item_idx] = optimizer.ga.shelf_locations[slot_idx]
        
    sorted_locs = [item_locations[i] for i in range(optimizer.num_items)]
    
    grid = np.zeros(optimizer.grid_size)
    
    for i in range(1, optimizer.num_items + 1):
        r, c = sorted_locs[i-1]
        
        freq = 0
        for order in optimizer.all_orders:
            if (i - 1) in order:  
                freq += 1
        
        grid[r, c] += freq
        
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap="YlOrRd", interpolation='nearest', aspect='auto')
    plt.colorbar(label='Pick Frequency / Traffic Estimate')
    plt.title("Warehouse Traffic Heatmap (Routing Congestion Analysis)")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/heatmap.png", dpi=300)
    print("Saved heatmap to results/heatmap.png")
    plt.close()

def run_sensitivity_analysis():
    print("Running Sensitivity Analysis: Warehouse Sizes")
    sizes = [(10, 10), (12, 12), (15, 15)]
    results = []
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "synthetic_warehouse_orders.csv")
    
    for size in sizes:
        print(f"Testing Grid Size: {size}")
        start_t = time.time()
        opt = HybridOptimizer(dataset_path=dataset_path, grid_size=size)
        pop = opt.ga.initialize_population()
        # Fast test: 5 generations
        best_fitness = float('inf')
        for g in range(5):
            pop, fitnesses = opt.run_generation(pop)
            best_fitness = min(best_fitness, min(fitnesses))
        end_t = time.time()
        
        results.append({
            'Size': f"{size[0]}x{size[1]}",
            'Runtime (s)': end_t - start_t,
            'Best Fitness': best_fitness
        })
        
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.bar(df['Size'], df['Runtime (s)'], color='skyblue')
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Runtime (s) [5 gens]')
    ax1.set_title('Scalability: Runtime vs Grid Size')
    
    ax2.plot(df['Size'], df['Best Fitness'], color='red', marker='o', linestyle='-', linewidth=2)
    ax2.set_xlabel('Grid Size')
    ax2.set_ylabel('Best Fitness Cost')
    ax2.set_title('Solution Quality vs Grid Size')
    
    plt.tight_layout()
    plt.savefig('results/scalability_analysis.png', dpi=300)
    print("Saved results/scalability_analysis.png")
    plt.close()

def run_product_mix_comparison():
    print("Running Product Mix Comparison (Uniform vs Pareto)")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uniform_path = os.path.join(base_dir, "dataset", "synthetic_warehouse_orders.csv")
    pareto_path = os.path.join(base_dir, "dataset", "pareto_warehouse_orders.csv")
    
    if not os.path.exists(pareto_path):
        print("Pareto dataset not found! Skipping mix comparison.")
        return

    datasets = {
        "Uniform Distribution": uniform_path,
        "80/20 Pareto Distribution": pareto_path
    }
    
    plt.figure(figsize=(8, 5))
    
    for label, path in datasets.items():
        print(f"Evaluating {label}...")
        set_global_seed(42)
        opt = HybridOptimizer(dataset_path=path)
        opt.ga.population_size = 20
        pop = opt.ga.initialize_population()
        
        convergence = []
        for g in range(50): # Increased to 50 generations to show learning curve
            pop, fitnesses = opt.run_generation(pop)
            convergence.append(min(fitnesses))
            
        plt.plot(range(1, 51), convergence, marker='o', markersize=3, label=label)
        
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Cost = Distance + Time + Penalties)')
    plt.title('Convergence Comparison: Uniform vs Pareto Product Mix')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/product_mix_comparison.png', dpi=300)
    print("Saved results/product_mix_comparison.png")
    plt.close()

def plot_operator_comparison():
    print("Running Operator Comparison Plot")
    stats_path = "results/batch_30_seed_stats.csv"
    if not os.path.exists(stats_path):
        print(f"Warning: {stats_path} not found. Skipped operator comparison plot.")
        return
        
    df = pd.read_csv(stats_path)
    
    plt.figure(figsize=(10, 6))
    df.boxplot(column='BestFitness', by='Operator', grid=False, rot=15)
    plt.title('Operator Performance Comparison (30 Seeds)')
    plt.suptitle('')  # Removes the default pandas subtitle
    plt.ylabel('Best Fitness Cost')
    plt.xlabel('Operator Combination (Selection_Mutation)')
    
    plt.tight_layout()
    plt.savefig('results/operator_comparison.png', dpi=300)
    print("Saved results/operator_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("Executing final reporting scripts...")
    set_global_seed(42)
    # Use one of the datasets to plot a heatmap
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "synthetic_warehouse_orders.csv")
    
    opt = HybridOptimizer(dataset_path=dataset_path)
    pop = opt.ga.initialize_population()
    
    plot_heatmap(opt, pop[0])
    run_sensitivity_analysis()
    run_product_mix_comparison()
    plot_operator_comparison()
