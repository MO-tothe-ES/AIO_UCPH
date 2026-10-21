# Project 1

Open `Simple_assignment.ipynb` and choose Run All. Keep `data/cars.csv`
in the data subfolder beside the notebook. Standard package dependencies
are listed in `requirements.txt`.

The notebook is self-contained: it estimates simple logit OLS/IV and
class-nested logit OLS/IV, computes elasticities and diversion, then simulates
VW-Fiat in every market and plots country effects. It reports one welfare
measure and one cost-saving threshold. The final two optional blocks
reproduce robustness and the assignment's theory test.

Both IV models use the same two counts: other own-manufacturer cars and
rival-manufacturer cars in the same class. Only price is logged; controls
are weight, power, fuel and home, with additive country and brand effects.
Results are calculated from the data and written to `output/`.
