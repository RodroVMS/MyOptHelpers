from typing import Tuple
from .utils import display_table, is_feasible_dual_simplex, update_table

# Table:
# Xb | x1 | ... | xn | y0
#  .
#  .
# rj | r1 | ... | rn |-z0

def dual_simplex(table, base, display=True, slacks=0):
    if not is_feasible_dual_simplex(table):
        print("Non dual-feasible solution")
        return table, base, False
    
    out_vect = out_vector(table)
    if out_vect == -1:
        print("Optimum Found")
        return table, base, True
    
    next_vect = next_vector(table, base, out_vect)
    if next_vect == -1:
        print("Unbounded Problem")
        return table, base, False

    dummy = base[out_vect]

    table, base = update_table(table, base, next_vect, out_vect)
    if display:
        display_table(table, base, next_vect, dummy, slacks)
    
    return dual_simplex(table, base, display, slacks)


def next_vector(table, base, out_vect):
    row = table[out_vect]
    min_ind = -1
    min_val = 2**32
    for i, val_i in enumerate(row[:-1]):
        if i in base or val_i >= 0:
            continue
        
        ri = table[-1][i]
        new_val = -ri/val_i
        if new_val < min_val:
            min_val = new_val
            min_ind = i
    
    return min_ind

def out_vector(table):
    min_ind = -1
    min_val = 2**32
    for i, row in enumerate(table[:-1]):
        val = row[-1]
        if val < 0 and val < min_val:
            min_val = val
            min_ind = i
    
    return min_ind
