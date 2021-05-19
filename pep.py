from utils import add_condition_to_table, display_table, get_int, is_feasible_dual_simplex, is_feasible_simplex, update_table
from simplex import next_vector, out_vector
from dual_simplex import dual_simplex

def pep(table, base, display=True, slacks=0):
    if not is_feasible_simplex(table):
        print("Unfeasible solution.")
        return table, base, False

    next_vect = next_vector(table, base)
    if next_vect == -1:
        print("Optimum found")
        return table, base, True
    
    out_vect = out_vector(table, next_vect)
    if out_vect == -1:
        print("Problem is unbounded")
        return table, base, False
    
    dummy = base[out_vect]

    if table[out_vect][next_vect] == 1:
        new_table, new_base = update_table(table, base, next_vect, out_vect)
        if display:
            display_table(new_table, new_base, slacks=slacks)
    else:
        condition = get_condition(table, base, next_vect, out_vect)
        new_table, new_base, slacks = add_condition_to_table(table, base, condition, slacks)
        if display:
            print("Adding condition.")
            display_table(new_table, new_base, slacks=slacks)
        if is_feasible_dual_simplex(table):
            print("Re-Optimizing with dual-simplex.")
            new_table, new_base, result = dual_simplex(new_table, new_base, display, slacks)
    
    return pep(new_table, new_base, display, slacks)

def get_condition(table, base, in_vect, out_vect):
    pivot = table[out_vect][in_vect]
    row = table[out_vect]
    new_cond = []
    for i, val in enumerate(row):
        if i == len(row) - 1:
            new_cond.append(1)
        if i in base:
            new_cond.append(0)
        else:
            new_cond.append(get_int(val/pivot))
    return new_cond