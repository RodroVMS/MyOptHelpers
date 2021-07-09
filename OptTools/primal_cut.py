from .utils import add_condition_to_table, display_table, get_int, is_feasible_dual_simplex, is_feasible_simplex, update_table
from .simplex import next_vector
from .dual_simplex import dual_simplex

def primal_cut(table, base, slacks, display):
    if not is_feasible_simplex(table):
        print("Unfeasible solution.")
        return table, base, False
    
    if is_feasible_dual_simplex(table):
        print("Optimal solution found (Dual Factible)")
        return table, base, True

    next_vect = next_vector(table, base)
    if next_vect == -1:
        print("Optimum found(Did not found next vector)")
        return table, base, True
    
    out_vect = out_vector(table, next_vect)
    if out_vect == -1:
        print("Problem is unbounded")
        return table, base, False
    
    dummy = base[out_vect]

    if table[out_vect][next_vect] == 1:
        new_table, new_base = update_table(table, base, next_vect, out_vect)
        if display:
            display_table(new_table, new_base, next_vect, dummy, slacks=slacks)
    else:
        condition = get_condition(table, base, next_vect, out_vect)
        new_table, new_base, slacks = add_condition_to_table(table, base, condition, slacks)
        if display:
            print("Adding condition.")
            display_table(new_table, new_base, slacks=slacks)
    
    return primal_cut(new_table, new_base, slacks, display)

def out_vector(table, col):
    out_val = 2**32
    pos_out = []
    for i, row in enumerate(table[:-1]):
        val_ij, y_0i = row[col], row[-1]

        if val_ij <= 0:
            continue

        new_val = y_0i/val_ij
        if new_val < out_val:
            out_val = new_val
            pos_out = [(i, val_ij)]
        elif new_val == out_val:
            pos_out.append((i, val_ij))
    
    out_ind = sorted(pos_out, key = lambda x: x[1])[0][0]
    return out_ind

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