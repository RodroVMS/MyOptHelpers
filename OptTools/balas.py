from typing import List, Tuple
import numpy as np
from collections import deque
from itertools import product


MAX_VAL = 2**32 - 1
MIN_VAL = (-1)*MAX_VAL

def balas(c:np.ndarray, A:np.ndarray, b:np.ndarray):
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
    var_num = A.shape[1]
    z_e = MIN_VAL
    
    x_0:List[int] = [0]*var_num
    w_0_0:Tuple[int, ...] = tuple()
    w_0_1:Tuple[int, ...] = tuple()
    pending = [(x_0, w_0_0, w_0_1)]
    while len(pending) > 0:
        x_k, w_k_0, w_k_1 = pending.pop(-1) 

        w_k = (*w_k_0, *w_k_1)
        L_k = [i for i in range(var_num) if not i in w_k]

        z_k = np.sum(c[w_k_1])
        if z_k > z_e:
            z_e = z_k
        else:
            continue

        s_k = get_sk(A, b, w_k_1)
        if np.all(s_k >= 0):
            continue

        Q_k = [i for (i, s_k_i) in enumerate(s_k) if s_k_i > 0]
        for i in Q_k:
            t_i = sum(min(0, A[i, j]) for j in L_k)
            if t_i > s_k[i]:
                continue
        
        R_k = []
        for i in Q_k:
            for j in L_k:
                if A[i,j] < 0:
                    R_k.append(i)
                    break
        
        p = A.shape[0]
        I_k_p = MAX_VAL
        for j in R_k:
            I_k_j = sum(min(0, (s_k[i] - A[i, j])) for i in Q_k)
            if I_k_j < I_k_p:
                p = j
                I_k_p = I_k_j
        
        w_k1_0 = tuple(sorted([i for i in w_k_0] + [p]))
        x_k1_0 = x_k.copy()
        x_k1_0[p] = 0
        pending.append((x_k1_0, w_k1_0, w_k_0))

        w_k2_1 = tuple(sorted([i for i in w_k_1] + [p]))
        x_k2_1 = x_k.copy()
        x_k2_1[p] = 1
        pending.append((x_k2_1, w_k_0, w_k2_1))

def get_sk(A, b, w_k_1) -> np.ndarray: 
    A_k_1 =  A[:, w_k_1]
    return b - np.sum(A_k_1, axis=(1,))

