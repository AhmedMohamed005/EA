## Results: Operator Comparison

"Table X presents descriptive statistics across 30 independent runs for each operator configuration. Mean best fitness ranged from 452.00 (roulette_inversion) to 455.00 (tournament_inversion), with standard deviations between 15.09 and 18.95. Independent-samples t-tests revealed no statistically significant difference between any pair of configurations (all p > 0.05). This indicates that, under the tested parameter regime of 20 generations and population size 20, the GA-ACO framework's convergence is primarily driven by the ACO routing component rather than the choice of GA operator. Future work could investigate operator sensitivity at higher generation counts or larger population sizes."

## Limitations & Discussion Notes

**1. Partial Result Correlation**
Partial result correlation was observed across operator configurations sharing the same seed (e.g., 13 out of 30 seeds produced identical results for tournament_swap vs tournament_inversion). This is attributed to the stochastic nature of the coin-flip crossover selection masking downstream mutation variance when operating on identical starting populations.

**2. Framework Sensitivity**
No statistically significant difference was found between operators at α = 0.05 (p > 0.58 for all comparisons). This suggests the hybrid GA-ACO framework's fitness landscape, dominated by the ACO routing component, is largely insensitive to the choice of GA selection and mutation operator under the tested parameter regime.

**3. O(N²) Linear Scanning Overhead**
The `_crowding_replacement` diversity mechanism forces an un-optimized pairwise calculation. This significantly increases execution time and overhead when `pop_size` scales exponentially higher.

**4. Congestion Routing Context**
The current computational model evaluates edges across multiple ant "traces", but this remains heavily disconnected from time dimensions. Pathing penalties are applied identically whether 15 simultaneous pickers enter an aisle at minute 1 versus minute 60. Sequential time-based constraints must be explicitly evaluated for real-world accuracy.

**5. Artificial Soft Surcharges**
An arbitrary "+15.0" operational multiplier anchors invalid packing bounds at 150 points. This effectively serves as a static penalization floor, skewing the fitness evaluation of initial generations.
