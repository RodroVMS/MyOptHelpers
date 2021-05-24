from ..OptTools import gomori_pi, pep, simplex, display_table
import numpy as np

table_1 = np.array(
    [
        [ 2,  2,  0, 1, 0, 3],
        [ 1,  2,  1, 0, 1, 2],
        [-3, -4, -1, 0, 0, 6]
    ],
    dtype="float64"
)
base_1 = [3, 4]
slacks = 3

table_3 = np.copy(table_1)
base_3 = base_1.copy()

print("Applying Gomori")
print("First Phase")
display_table(table_1, base_1, slacks=slacks)
table_1, base_1, result =  simplex(table_1, base_1, True, slacks)

print("Second Phase")
slacks = 1
table_2 = np.array([np.concatenate((row[:3], row[-1:])) for row in table_1])
table_2[-1][-2] = -1 
table_2[-1][-1] = 0
base_2 = base_1

display_table(table_2, base_2, slacks=slacks)
table_2, base_2, result = gomori_pi(table_2, base_2, True, slacks)

print("\nApplying PEP")
print("First Phase")
display_table(table_3, base_3, slacks=3)
table_3, base_3, result = pep(table_3, base_3, True, slacks=3)

print("First Phase minimum value is non zero due to s2 != 0. The problem cannot be solved")