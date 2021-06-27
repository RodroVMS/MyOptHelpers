from OptTools.balas import balas
from OptTools.utils import get_float64_ndarray
import sys
import numpy as np
from OptTools import display_table, simplex, gomori_cut, primal_cut, get_basic_solution

def main(arg):
    if arg in {"ex1", "ej1", "1"}:
        example1()
    if arg in {"ex2", "ej2", "2"}:
        example2()
    if arg in {"ex3", "ej3", "3"}:
        example3()

def example1():
    table_1 = np.array(
        [
            [1,  3, 1, 0, 5],
            [2,  1, 0, 1, 6],
            [2, -3, 0, 0, 0]
        ],
        dtype="float64"
    )
    base_1 = [2, 3]
    slacks = 2
    
    table_2 = np.copy(table_1)
    base_2 = base_1.copy()
    
    print("Solving with Gomori")
    display_table(table_1, base_1, slacks=slacks)
    table_1, base_1, result = gomori_cut(table_1, base_1, slacks, True)
    print(get_basic_solution(table_1, base_1))
    
    print("\nSolving with pep")
    display_table(table_2, base_2, slacks=slacks)
    table_2, base_2, result = primal_cut(table_2, base_2, slacks, True)
    print(get_basic_solution(table_2, base_2))

def example2():
    table_1 = np.array(
        [
            [ 2,  2,  0, 1, 0, 3],
            [ 1,  2,  1, 0, 1, 2],
            [-3, -4, -1, 0, 0, 6]
        ],
        dtype="float64"
    )
    slacks = 3

    table_1 = np.array(
        [
            [ 2,  2,  1,  0, 3],
            [ 1,  2,  0,  1, 2],
            [-2, -2,  0,  0, 6]
        ],
        dtype="float64"
    )
    slacks = 2
    base_1 = [2, 3]


    table_3 = np.copy(table_1)
    base_3 = base_1.copy()

    print("Applying Gomori")
    print("First Phase")
    display_table(table_1, base_1, slacks=slacks)
    table_1, base_1, result =  simplex(table_1, base_1, slacks, True)

    print("Second Phase")
    slacks = 1
    table_2 = np.array([np.concatenate((row[:2], row[-2:])) for row in table_1])
    table_2[-1][1] = 1 
    table_2[-1][-1] = 0
    base_2 = [0, 2]

    display_table(table_2, base_2, slacks=slacks)
    table_2, base_2, result = gomori_cut(table_2, base_2, slacks, True)

    print("\nApplying PEP")
    print("First Phase")
    display_table(table_3, base_3, slacks=2)
    table_3, base_3, result = primal_cut(table_3, base_3, slacks=2, display=True)
    print(get_basic_solution(table_3, base_3))
    print("First Phase minimum value is non zero due to s2 != 0. The problem cannot be solved")

def example3():
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
    table_1, base_1, result =  primal_cut(table_1, base_1, slacks, True)

    print("Second Phase")
    table_2 = np.array([np.concatenate((row[:3], row[-3:])) for row in table_1])
    table_2[-1, 1] = 14
    table_2[-1, -1] = 3

    base_2 = base_1
    base_2[0] -= 2
    slacks = 3

    display_table(table_2, base_2, slacks=slacks)
    table_2, base_2, result = primal_cut(table_2, base_2, slacks, display=True)
    print(get_basic_solution(table_2, base_2))

def example4():
    c = get_float64_ndarray(
        [-5, -7, -10, -3, -1]
    )
    A = get_float64_ndarray(
        [
            [-1,  3, -5, -1,  4],
            [ 2, -6,  3,  2, -2],
            [ 0,  1, -2,  1,  1]
        ]
    )
    b = get_float64_ndarray(
        [-2, 0, -1]
    )

    result, x_e, z_e = balas(c, A, b, display=True)

# if __name__ == "__main__":
#     args = sys.argv[1:]
#     main(*args)

example4()

