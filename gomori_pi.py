from dual_simplex import dual_simplex
from simplex import simplex
from utils import add_condition_to_table, display_table, get_decimals, is_feasible_dual_simplex, is_feasible_simplex, pure_integer_vector


def gomori_pi(table, base, display = True, slacks = 0):
    result = False
    if is_feasible_simplex(table):
        print("Minimizing with simplex")
        table, base, result = simplex(table, base, display, slacks)
    elif is_feasible_dual_simplex(table):
        print("Minimizing with dual-simplex")
        table, base, result = dual_simplex(table, base, display, slacks)
    
    if not result:
        print("Could not minimize")
        return table, base, False

    if pure_integer_vector(table[:,-1][:-1]):
        print("Optimum Found")
        return table, base, True
    
    row_list = get_condition_rows(table, base)
    row = select_greater(row_list)

    new_table, new_base, slacks = add_condition_to_table(table, base, row, slacks)
    if display:
        print("Adding new condition")
        display_table(new_table, new_base, slacks=slacks)
    
    print("Re-Optimizing with Dual-Simplex")
    new_table, new_base, result = dual_simplex(new_table, new_base, display, slacks)
    
    return gomori_pi(new_table, new_base, display, slacks)


def get_condition_rows(table, base):
    row_list = []
    for row in table:
        new_row = []
        for i, val in enumerate(row):
            if i == len(row) - 1:
                new_row.append(1)
            new_row.append(-get_decimals(val)) if not i in base else new_row.append(0)
        row_list.append(new_row)
    
    return row_list
    
def select_greater(row_list):
    max_ind = len(row_list)
    max_val = -2**31 
    for i, row in enumerate(row_list):
        if 0 < row[-1]*(-1) and row[-1]*(-1) > max_val:
            max_val = row[-1]*(-1)
            max_ind = i
    
    return row_list[max_ind]
