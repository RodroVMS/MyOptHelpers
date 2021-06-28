from typing import List, Tuple
import numpy as np
from .const import MAX_VAL, MIN_VAL




def balas(c:np.ndarray, A:np.ndarray, b:np.ndarray, display:bool=False):
    """
    #### Balas Algorithm:
    Applies balas on a problem of the form:
    :max z = (c^t)
    :s.t Ax <= b
    #### Params:
    c: one dimensional array
    A: nxm matrix
    b: one dimensional array
    """
    cond_num = A.shape[0]
    var_num = A.shape[1]
    z_e = MAX_VAL
    x_e = [-1]*var_num
    
    output = ""
    max_node_num:int = int(0) # WTF Pylance !!!
    w_e = tuple()

    x_0:List[int] = [0]*var_num
    w_0_0:Tuple[int, ...] = tuple()
    w_0_1:Tuple[int, ...] = tuple()
    pending = [(x_0, w_0_0, w_0_1, max_node_num)]
    while len(pending) > 0:
        x_k, w_k_0, w_k_1, node_num = pending.pop(-1) 

        w_k = (*w_k_0, *w_k_1)
        L_k = [i for i in range(var_num) if not i in w_k]
        z_k = np.sum([c[i] for i in w_k_1])

        x_str = display_x(x_k, w_k)
        output += "\n-------------- -------------- --------------"  
        output += f"\nNode({node_num}):\nx: {x_str}\nz: {z_k}\nze: {z_e}"

        s_k = get_sk(A, b, w_k_1)
        if np.all(s_k >= 0):
            output += "\nx^ is factible. Found optimum solution for subproblem."
            
            if z_k < z_e:
                z_e = z_k
                x_e = x_k.copy()
                w_e = w_k
                output += f"\nUpdated ze: {z_e}"
            else:
                output += f"\nze not improved: {z_e} < {z_k}"
            continue
        else:
            output += "\nx^ is not factible."

        Q_k = [i for (i, s_k_i) in enumerate(s_k) if s_k_i < 0]
        if not factible(A, L_k, Q_k, s_k):
            output += "\nNode cannot be factible, nor it's descendants."
            continue
        
        R_k = []
        for j in L_k:
            for i in Q_k:
                if A[i,j] < 0:
                    R_k.append(j)
                    break

        output += f"\nQ_k: {Q_k}  R_k: {R_k}"
        
        p = A.shape[0]
        I_k_p = MAX_VAL
        for j in R_k:
            I_k_j = sum(min(0, (s_k[i] - A[i, j])) for i in range(cond_num))
            I_k_j = np.abs(I_k_j)
            
            output += f"\nI({j}) = {I_k_j}"
            if I_k_j < I_k_p:
                p = j
                I_k_p = I_k_j
        
        output += f"\np = {p}  Ik(p) = {I_k_p}"
        
        w_k1_0 = tuple(sorted([i for i in w_k_0] + [p]))
        x_k1_0 = x_k.copy()
        x_k1_0[p] = 0
        pending.append((x_k1_0, w_k1_0, w_k_1, max_node_num + 2))

        w_k2_1 = tuple(sorted([i for i in w_k_1] + [p]))
        x_k2_1 = x_k.copy()
        x_k2_1[p] = 1
        pending.append((x_k2_1, w_k_0, w_k2_1, max_node_num + 1))     

        max_node_num += 2
    
    output += "\n-------------- -------------- --------------" 
    if z_e == MIN_VAL:
        output += "\nUnfeasible solution"
        if display:
            print(output)
        return False, x_e, z_e
    
    if display:
        output += f"\nSolution:\nxe: {display_x(x_e, w_e)}\nze: {z_e}"
        print(output)
    return True, x_e, z_e

def factible(A, L_k, Q_k, s_k) -> bool:
    for i in Q_k:
        t_i = sum(min(0, A[i, j]) for j in L_k)
        if t_i > s_k[i]:
            return False
    return True


def get_sk(A, b, w_k_1) -> np.ndarray: 
    A_k_1 =  A[:, w_k_1]
    return b - np.sum(A_k_1, axis=(1,))

def display_x(x, w):
    disp_x = "("
    disp_x += ", ".join(f"{val}" if not i in w else f"{val}*" for i, val in enumerate(x))
    disp_x += ")"
    return disp_x