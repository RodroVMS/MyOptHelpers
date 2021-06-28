from MyOptHelpers.OptTools.const import MAX_VAL, MIN_VAL
import numpy as np
from .utils import add_condition_to_table, basic_solution_str, display_table, get_decimals, get_int, is_feasible_simplex, is_feasible_dual_simplex, get_basic_solution, pure_integer_value, pure_integer_vector, update_table, var_name
from .simplex import simplex
from .dual_simplex import dual_simplex


def land_doig(func, start_table:np.ndarray, start_base:list, start_slacks:int = 0, display:bool = True):
    total_node_num = int(0) # Pylance victim
    
    output = ""
    original_vars = len(start_table[0]) - start_slacks - 1
    x_e = [-1]*original_vars
    xs_e = [-1]*original_vars
    table_e = np.array([])
    base_e = []
    slacks_e = -1
    number_e = -1
    z_e = MAX_VAL

    pending = [(start_table.copy(), tuple(start_base), start_slacks, total_node_num, str(""))]
    while len(pending) > 0:
        table, tup_base, slacks, node_num, trail = pending.pop(-1)
        base = list(tup_base)
        print("\n-------------- -------------- --------------")
        print(f"Solving Node({node_num}): {trail}")
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
            print(f"No improvement in solution: zk({z_k}) >= ze({z_e})")
            continue

        if pure_integer_vector(x_k):
            x_e = [x_k[i] for i in range(min(len(x_k), original_vars))]
            xs_e = x_k.copy()
            z_e = z_k
            table_e = table
            base_e = base
            slacks_e = slacks
            number_e = node_num
            print(f"Pure integer found. Updating ze = {z_e}")
            continue
        
        index = len(table[0])
        max_deg = MIN_VAL
        degrad_str = ""
        for i, y_i in enumerate(table[:, -1]):
            if not pure_integer_value(y_i) and i < len(table) - 1:
                deg_i_neg = get_decimals(y_i)
                deg_i_plus = 1 - deg_i_neg
                deg_i = min(deg_i_neg, deg_i_plus)

                degrad_str += f"Deg({var_name(len(table[0]) -1, slacks, i)}) = {format(deg_i, '.4f')}, "
                if deg_i > max_deg:
                    index = i
                    max_deg = deg_i

        print(degrad_str)
        new_cond_le, new_cond_ge = form_conditon(table, base, index)
        table_le, base_le, slacks_le = add_condition_to_table(table, base, new_cond_le, slacks)
        table_ge, base_ge, slacks_ge = add_condition_to_table(table, base, new_cond_ge, slacks)

        print("Applying dual simplex-ge")
        # display_table(table_ge, base_ge, slacks=slacks_ge)
        # print("Updated:")
        table_ge, base_ge = update_table(table_ge, base_ge, base_ge[index], index)
        # display_table(table_ge, base_ge, slacks=slacks_ge)
        table_ge, base_ge, result = dual_simplex(table_ge, base_ge, slacks_ge, display=True)
        if result:  
            total_node_num += 1
            print(f"Added child({total_node_num})")
            pending.append((table_ge, tuple(base_ge), slacks_ge, total_node_num, trail + f"{var_name(len(table_ge[0]) -1, slacks_ge, index)} >= {np.floor(max_deg) + 1}, "))
        
        print("Applying dual simplex-le")
        # display_table(table_le, base_le, slacks=slacks_le)
        table_le, base_le =  update_table(table_le, base_le, base_le[index], index)
        # print("Updated:")
        # display_table(table_le, base_le, slacks=slacks_le)
        table_le, base_le, result = dual_simplex(table_le, base_le, slacks_le, display=False)
        if result:
            total_node_num += 1 
            print(f"Added child({total_node_num})")
            pending.append((table_le, tuple(base_le), slacks_le, total_node_num, trail + f"{var_name(len(table_le[0]) -1, slacks_le, index)} <= {np.floor(max_deg)}, "))
        
    print("\n-------------- -------------- --------------")
    if z_e == MAX_VAL:
        print("Problam has no factible solution")
        return table_e, base_e, False
    
    print(f"Solution found on Node({number_e}):")
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
