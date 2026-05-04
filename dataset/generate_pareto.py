# generate_pareto.py
import pandas as pd
import random
import numpy as np
import os

def generate_80_20_dataset(num_orders=100, num_items=100, output_path="pareto_warehouse_orders.csv"):
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_path)
        
    # 20% of items are picked 80% of the time.
    hot_items = np.arange(0, int(num_items*0.2))
    cold_items = np.arange(int(num_items*0.2), num_items)
    
    orders = []
    for _ in range(num_orders):
        order_size = random.randint(1, 5)
        # 80% chance for an item to come from hot_items
        order = []
        for _ in range(order_size):
            if random.random() < 0.8 and len(hot_items) > 0:
                order.append(f"ITEM_{random.choice(hot_items):03d}")
            else:
                order.append(f"ITEM_{random.choice(cold_items):03d}")
        
        # remove duplicates
        order = list(set(order))
        if order:
            orders.append({
                "OrderID": f"ORD_{len(orders):03d}",
                "ItemsList": ", ".join(order)
            })
            
    df = pd.DataFrame(orders)
    df.to_csv(output_path, index=False)
    print(f"Pareto 80/20 dataset saved to {output_path}")

if __name__ == "__main__":
    generate_80_20_dataset()
