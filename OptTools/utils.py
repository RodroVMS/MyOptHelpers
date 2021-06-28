import numpy as np
import pandas as pd

def update_table(table, base, in_vect, out_vect):
    base[out_vect] = in_vect

    if table[out_vect][in_vect] != 1:
        table[out_vect] = table[out_vect]/table[out_vect][in_vect]
    
    for i, row in enumerate(table):
        if i == out_vect or row[in_vect] == 0:
            continue
        val = row[in_vect]
        pivot_row = table[out_vect]*val
        table[i] = row - pivot_row
    
    return table, base

def add_condition_to_table(table:np.ndarray, base, new_condition, slacks = 0):
    new_table = np.full((table.shape[0] + 1, table.shape[1] + 1), 0.0, dtype="float64")

    for i, row in enumerate(table[:-1]):
        new_table[i] = np.concatenate((row[:-1], [0], row[-1:]))

    new_table[-2] = np.array(new_condition)
    new_table[-1] = np.concatenate((table[-1][:-1], [0], table[-1][-1:]))

    new_base = base + [len(table[0]) - 1] # New var is now base
    return new_table, new_base, slacks + 1

def display_table(table, base, next_vect = None, out_vect = None, slacks = 0):    
    d = {"Xb": [f"x{i + 1}" if i < len(table[0]) - 1 - slacks else f"s{i - len(table[0]) + slacks + 2}" for i in base] + ["rj"]}
    
    for row in table:
        for j in range(len(row)):
            if j == len(row) - 1:
                key = "y0"
            elif j >= len(row) - slacks - 1:
                key = f"s{j - len(row) + slacks + 2 }"
            else:
                key = f"x{j + 1}"

            try:
                d[key].append(format(row[j], ".2f"))
            except KeyError:
                d[key] = [format(row[j], ".2f")]
    
    df = pd.DataFrame(d)
    if next_vect is not None and out_vect is not None:
        next_name = f"x{next_vect + 1}" if next_vect < len(table[0]) -1 -slacks else f"s{next_vect - len(table[0]) + slacks + 2}"
        out_name = f"x{out_vect + 1}" if out_vect < len(table[0]) -1 -slacks else f"s{out_vect - len(table[0]) + slacks + 2}"
        print(f"In {next_name}. Out {out_name}.")
    
    #print(display_table_md(table, base, slacks))
    print(df.to_string(index=False), "\n")

def display_table_md(table, base, slacks = 0):
    total_vars = len(table[0]) - 1
    header = "| $X_b$ | " + " | ".join([var_name_md(total_vars, slacks, i) for i in range(total_vars)]) + " | $y_0$ |\n"
    header += "|" + "|".join(["----" for _ in range(total_vars + 2)]) + "|\n"
    for i, row in enumerate(table):
        row_str = ""
        for j, val in enumerate(row):
            if j == 0:
                if i == len(table) - 1:
                    row_str += "| $r_j$ | "
                else:
                    row_str += f"| {var_name_md(total_vars, slacks, base[i])} | "
            if i == len(table) - 1 and j == len(row) - 1:
                row_str += " |"
                continue
            row_str += (str(int(val)) if val == get_int(val) else format(val, ".2f")) + " | "
        header += row_str + "\n"
    return header

def var_name_md(total_var, slacks, i):
    if i < total_var - slacks:
        return r"$x_{"+ str(i + 1) +r"}$"
    else:
        return r"$s_{" + str(i - total_var + slacks + 1) + r"}$"


def is_feasible_simplex(table) -> bool:
    return not any(table[:, -1][:-1] < 0)

def is_feasible_dual_simplex(table) -> bool:
    return not any(table[-1][:-1] < 0)

def get_basic_solution(table, base):
    basic = []
    for i in range(len(table[0]) - 1):
        try:
            index = base.index(i)
            basic.append(table[:, -1][index])
        except ValueError:
            basic.append(0)
    return  basic #

def basic_solution_str(basic:list):
    return "(" + ", ".join(str(int(val)) if val == get_int(val) else format(val, ".2f") for val in basic) + ")"

def pure_integer_vector(y0):
    for val in y0:
        if not pure_integer_value(val):
            return False
    return True

def pure_integer_value(val):
    return val - np.floor(val) <= 1e-9

def get_decimals(val):
    return val - np.floor(val)

def get_int(val):
    return np.floor(val)

def get_float64_ndarray(l:list):
    return np.array(l, dtype="float64")