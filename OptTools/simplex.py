from .utils import is_feasible_simplex, update_table, display_table

# Table:
# Xb | x1 | ... | xn | y0
#  .
#  .
# rj | r1 | ... | rn |-z0

def simplex(table, base, slacks, display):
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
    table, base = update_table(table, base, next_vect, out_vect)
   
    if display:
        display_table(table, base, next_vect, dummy, slacks)
    
    return simplex(table, base, slacks, display)
    
def next_vector(table, base):
    minimum_ind = -1
    minimum_val = 2**32
    for j, rj in enumerate(table[-1][:-1]):
        if j in base:
            continue
        if rj < 0 and rj < minimum_val:
            minimum_val = rj
            minimum_ind = j

    return minimum_ind

def out_vector(table, col):
    out_ind = -1
    out_val = 2**32
    for i, row in enumerate(table[:-1]):
        val_ij, y_0i = row[col], row[-1]

        if val_ij <= 0:
            continue

        new_val = y_0i/val_ij
        if new_val < out_val:
          out_ind = i
          out_val = new_val
    
    return out_ind
