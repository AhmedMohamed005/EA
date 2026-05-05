# AI420 Master Project Plan: Smart Warehouse Layout Optimisation 
**Module:** AI420 Evolutionary Algorithms (Spring 2026)
**Project Idea:** [8] Smart Warehouse Layout Optimisation using Hybrid GA-ACO
**Team Size:** 5-6 Members

---

## 1. Problem Formalization & Architecture
To achieve the "Bonus" and build a robust solution, we are formulating this as a **Constrained Optimisation Problem** using a **Hybrid GA-ACO** approach.

### The Hybrid Concept
*   **Genetic Algorithm (GA):** Responsible for the **Macro-Layout**. It decides *where* the shelves, packing zones, and shipping docks go within the warehouse grid.
*   **Ant Colony Optimisation (ACO):** Responsible for the **Micro-Routing**. Once the GA proposes a layout, the ACO simulates workers (ants) picking a batch of orders to calculate the exact travel distance.
*   **Fitness Function:** The total distance/time calculated by the ACO becomes the "fitness score" for the GA's layout. Lower score = Better fitness.

### Constraints Handling
*   **Space Constraints:** Shelves cannot overlap with walls or each other.
*   **Safety Constraints:** Hazardous materials must be placed in specific zones (simulated via **Penalty Functions**—if the GA places them wrong, the fitness score is artificially penalized/increased).

---

## 2. Team Roles & Responsibilities (6 Members)
*All members must contribute to the code and the report. This division ensures accountability.*

1.  **Lead Architect / Integrator:** Manages the GitHub repository, integrates GA with ACO, and handles the random seed management.
2.  **GA Developer:** Implements the layout representation, initialisation, crossover, mutation, and selection operators.
3.  **ACO Developer:** Implements the grid graph, pheromone matrix, heuristic visibility, and evaporation logic.
4.  **GUI & Simulation Engineer:** Builds the visual interface (e.g., using PyQt or Tkinter) to animate the warehouse layout and the ants moving through aisles.
5.  **Data & Experiments Analyst:** Manages the 30 evaluation runs, statistical analysis, plotting convergence curves, and comparing the operators.
6.  **Research Lead (Literature & Docs):** Sources the 4-6 academic papers, writes the literature review, and structures the final academic report.

---

## 3. Algorithmic Requirements (Course Guidelines Checklist)

### A. Genetic Algorithm Components (Layout Generation)
*   **Representation:** 2D Grid / Matrix encoding mapping items to locations.
*   **Initialisation:** Random feasible generation (ensuring no overlapping shelves).
*   **Parent Selection (Implement at least 2):**
    1.  Tournament Selection ($k$-way)
    2.  Roulette Wheel Selection (Fitness Proportionate)
*   **Recombination / Crossover (Implement at least 2):**
    1.  Order Crossover (OX1) - useful for permutation of items.
    2.  Partially Mapped Crossover (PMX).
*   **Mutation (Implement at least 2):**
    1.  Swap Mutation (Swapping two shelf locations).
    2.  Inversion Mutation (Reversing a sequence of items in an aisle).
*   **Survivor Selection (Implement at least 2):**
    1.  Elitism (Keep top $N$ layouts).
    2.  Generational Replacement.
*   **Diversity Preservation:** Fitness Sharing or Crowding to prevent the GA from converging on a single layout too early.

### B. Ant Colony Optimisation Components (Routing Evaluation)
*   **Pheromone Matrix ($\tau$):** Tracks the "scent" between specific nodes (shelves/aisles).
*   **Heuristic Visibility ($\eta$):** The inverse of the distance between two nodes ($1/d$).
*   **Transition Rule:** The probability equation combining $\tau$ and $\eta$ to decide the ant's next step.
*   **Evaporation:** Reducing pheromones every generation by a factor $\rho$ to prevent getting stuck in local optima.

---

## 4. Experiments & Data Protocol

### The Dataset
*   **Public Dataset:** Use a benchmark TSP or Warehouse dataset (e.g., from OR-Library or Kaggle) representing item frequencies/order batches.
*   **Simulated Dataset:** Create a synthetic dataset of 1000 orders, where 20% of items appear in 80% of orders (Pareto principle) to test if the algorithm groups popular items near the shipping dock.

### The 30-Run Rule
To prove the algorithm's statistical significance (mandatory for the course):
1.  Define a fixed list of 30 integer seeds (e.g., `seeds = [42, 101, 202, ... 3030]`).
2.  Run the *entire* Hybrid Algorithm 30 times for **each** setting combination.
    *   *Experiment 1:* GA with Tournament vs. GA with Roulette.
    *   *Experiment 2:* GA with Swap Mutation vs. Inversion Mutation.
3.  Log the Best, Worst, Average, and Standard Deviation of the fitness scores for each run.

---

## 5. Development Sprints (Timeline)

*   **Sprint 1 (Setup):** Finalize problem formulation. Gather dataset. Set up GitHub and Python environment.
*   **Sprint 2 (Core Logic):** Build the basic GA (Layouts) and basic ACO (Routing) independently.
*   **Sprint 3 (Integration & Constraints):** Connect GA and ACO. Apply the Penalty Functions for constraints. Implement the secondary crossover/mutation operators.
*   **Sprint 4 (GUI & Simulation):** Build the interface. It must show: Inputs (parameters), Live Simulation (layout changes), and Outputs (graphs).
*   **Sprint 5 (Experiments):** Lock the code. Run the 30-seed batches. Extract data to CSV and generate `matplotlib` charts.
*   **Sprint 6 (Documentation):** Finalize the report. Cross-check against the rubric.

---

## 6. Final Report Structure (Week 13/14 Submission)

1.  **Title Page:** Project title, Group members, Roles.
2.  **Introduction & Problem Definition:** Detailed explanation of the Smart Warehouse Layout problem and objectives.
3.  **Literature Review:** Summary of 4-6 recent academic papers solving warehouse routing/layouts using EAs.
4.  **Dataset Description:** Details of the employed dataset (public or synthetic) and order frequencies.
5.  **Methodology & Algorithms:**
    *   Formal mathematical representation.
    *   Detailed explanation of the GA (Representation, Operators, Diversity).
    *   Detailed explanation of the ACO.
    *   Explanation of the Hybridization and Constraint Handling.
6.  **Experiments & Results:**
    *   Parameter tuning tables.
    *   Convergence graphs.
    *   Comparative analysis of the different operators (30 runs data).
7.  **System Design & GUI:** Screenshots of the interface and architecture diagrams.
8.  **Conclusion & Future Work:** What worked, what didn't, and how it could be improved.
9.  **References:** IEEE or APA format for the 4-6 papers.