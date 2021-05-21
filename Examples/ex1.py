from ..OptTools import gomori_pi, pep, display_table
import numpy as np


table_1 = np.array(
    [
        [1,  3, 1, 0, 5],
        [2,  1, 0, 1, 6],
        [2, -3, 0, 0, 0]
    ]
)
base_1 = [2, 3]
slacks = 2

table_2 = np.copy(table_1)
base_2 = base_1.copy()

print("Solving with Gomori")
display_table(table_1, base_1, slacks=slacks)
table_1, base_1, result = gomori_pi(table_1, base_1, True, slacks)

print("\nSolving with pep")
display_table(table_2, base_2, slacks=slacks)
table_2, base_2, result = pep(table_2, base_2, True, slacks)