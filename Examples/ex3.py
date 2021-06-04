from ..OptTools import pep, display_table, get_basic_solution
import numpy as np

table_1 = np.array(
    [
        [ 2,   3,  0, 1, 0,  6],
        [ 2,   9,  1, 0, 1,  6],
        [-4, -12, -1, 0, 0, 12]
    ],
    dtype="float64"
)
base_1 = [3, 4]
slacks = 3

print("First Phase")
display_table(table_1, base_1, slacks=slacks)
table_1, base_1, result =  pep(table_1, base_1, True, slacks)

print("Second Phase")
table_2 = np.array([np.concatenate((row[:3], row[-3:])) for row in table_1])
table_2[-1, 1] = 14
table_2[-1, -1] = 3

base_2 = base_1
base_2[0] -= 2
slacks = 3

display_table(table_2, base_2, slacks=slacks)
table_2, base_2, result = pep(table_2, base_2, True, slacks)
print(get_basic_solution(table_2, base_2))
