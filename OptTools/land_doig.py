from MyOptHelpers.OptTools.const import MAX_VAL, MIN_VAL
import numpy as np
from .utils import add_condition_to_table, basic_solution_str, display_table, get_int, is_feasible_simplex, is_feasible_dual_simplex, get_basic_solution, pure_integer_value, pure_integer_vector, update_table
from .simplex import simplex
from .dual_simplex import dual_simplex


def land_doig(func, start_table:np.ndarray, start_base:list, start_slacks:int = 0, display:bool = True):
    total_node_num = int(0) # Pylance victim
    
    output = ""
    original_vars = len(start_table[0])
    x_e = [-1]*original_vars
    xs_e = [-1]*original_vars
    table_e = np.array([])
    base_e = []
    slacks_e = -1
    z_e = MAX_VAL

    pending = [(start_table.copy(), tuple(start_base), start_slacks, total_node_num)]
    while len(pending) > 0:
        table, tup_base, slacks, node_num = pending.pop(-1)
        base = list(tup_base)
        print("-------------- -------------- --------------")
        print(f"Solving Node{node_num}")
        display_table(table, base, slacks=slacks)


        result = False
        if is_feasible_simplex(table):
            print("Minimizing with simplex")
            table, base, result = simplex(table, base, slacks, display)
        elif is_feasible_dual_simplex(table):
            print("Minimizing with dual-simplex")
            table, base, result = dual_simplex(table, base, slacks, display)

        if result == False:
            print("Could not Minimize")
            continue

        x_k = get_basic_solution(table, base)
        z_k = func(x_k)

        print(f"xk: {basic_solution_str(x_k)}")
        print(f"zk: {z_k}")

        if z_k >= z_e:
            print(f"No improvement in solution: zk({z_k}) >= z_e({z_e})")
            continue

        if pure_integer_vector(x_k):
            x_e = [x_k[i] for i in range(min(len(x_k), original_vars))]
            xs_e = x_k.copy()
            z_e = z_k
            table_e = table
            base_e = base
            slacks_e = slacks
            print(f"Pure integer found. Updating z_e = {z_e}")
            continue

        for i, y_i in enumerate(table[:, -1]):
            if not pure_integer_value(y_i):
                new_cond_le, new_cond_ge = form_conditon(table, base, i)
                table_le, base_le, slacks_le = add_condition_to_table(table, base, new_cond_le, slacks)
                table_ge, base_ge, slacks_ge = add_condition_to_table(table, base, new_cond_ge, slacks)

                table_le, base_le =  update_table(table_le, base_le, base_le[i], i)
                print("Applying dual simplex-le")
                table_le, base_le, result = dual_simplex(table_le, base_le, slacks_le, display=False)
                if result:
                    pending.append((table_le, tuple(base_le), slacks_le, total_node_num + 2))

                print("Applying dual simplex-ge")
                table_ge, base_ge = update_table(table_ge, base_ge, base_ge[i], i)
                table_ge, base_ge, result = dual_simplex(table_ge, base_ge, slacks_ge, display=True)
                if result:   
                    pending.append((table_ge, tuple(base_ge), slacks_ge, total_node_num + 1))
                
                total_node_num += 2
                break
        
    print("-------------- -------------- --------------")
    if z_e == MAX_VAL:
        print("Problam has no factible solution")
        return table_e, base_e, False
    
    print("Solution found:")
    print(f"ze: {z_e}")
    print(f"xe: {x_e}")
    print(f"xse: {xs_e}")
    display_table(table_e, base_e, slacks=slacks_e)
    return table_e, base_e, True
                

def form_conditon(table, base, index):
    assert not pure_integer_value(table[index, -1]) 

    int_val = get_int(table[index, -1])
    var_index = base[index]

    new_cond_ge = [0]*(len(table[0]) + 1)
    new_cond_le = [0]*(len(table[0]) + 1)

    new_cond_le[var_index] =  1
    new_cond_ge[var_index] = -1

    new_cond_le[-2] = new_cond_ge[-2] = 1
    new_cond_le[-1] = int_val
    new_cond_ge[-1] = -(int_val + 1)

    return new_cond_le, new_cond_ge
