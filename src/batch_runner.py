import numpy as np
import pandas as pd
import os
from .hybrid import HybridOptimizer, set_global_seed

def run_batch_experiments(num_seeds=30, generations=20, pop_size=20, output_dir="results"):
    """
    Runs the mandatory 30-seed automated batch experiments for the course.
    Evaluates different genetic operators cleanly without GUI overhead.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Fixed reproducible seeds for exactly 30 runs
    seeds = list(range(1, num_seeds + 1))
    
    # Required Operator Comparison Combinations
    operator_configs = [
        {"selection": "tournament", "mutation": "swap"},
        {"selection": "roulette", "mutation": "swap"},
        {"selection": "tournament", "mutation": "inversion"},
        {"selection": "roulette", "mutation": "inversion"}
    ]
    
    experiment_results = []
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "synthetic_warehouse_orders.csv")
    
    for config in operator_configs:
        operator_name = f"{config['selection']}_{config['mutation']}"
        print(f"\n===========================================")
        print(f"--- Running Configuration: {operator_name} ---")
        print(f"===========================================\n")
        
        for seed in seeds:
            print(f"[{operator_name}] Seed {seed}/{num_seeds}...", end=" ")
            set_global_seed(seed)
            
            optimizer = HybridOptimizer(dataset_path=dataset_path)
            optimizer.ga.population_size = pop_size
            
            population = optimizer.ga.initialize_population()
            
            best_fitness_over_time = []
            best_overall = float("inf")
            
            for g in range(generations):
                new_pop, fitnesses = optimizer.run_generation(
                    population,
                    selection_method=config["selection"],
                    mutation_method=config["mutation"]
                )
                population = new_pop
                
                gen_best = min(fitnesses)
                best_overall = min(best_overall, gen_best)
                best_fitness_over_time.append(gen_best)
                
            print(f"Best: {best_overall:.2f}")
            
            experiment_results.append({
                "Operator": operator_name,
                "Seed": seed,
                "BestFitness": best_overall,
                "Convergence": best_fitness_over_time
            })
            
    # Compile statistics
    df_results = pd.DataFrame([{
        "Operator": res["Operator"],
        "Seed": res["Seed"],
        "BestFitness": res["BestFitness"]
        } for res in experiment_results
    ])
    
    print("\n--- Final Statistics over 30 Runs By Operator ---")
    stats = df_results.groupby('Operator')['BestFitness'].agg(['mean', 'std', 'min', 'max']).reset_index()
    print(stats.to_string(index=False))
        
    # Export raw data to CSV for the report plotting
    csv_path = os.path.join(output_dir, "batch_30_seed_stats.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nStats exported to {csv_path}")

if __name__ == "__main__":
    run_batch_experiments()
