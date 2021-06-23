import numpy as np
from OptTools import simplex, dual_simplex, gomori_pi, pep

def optimize(method:str, inital_table:np.ndarray, usr_options:dict = dict()):
    options = set_options(usr_options)
    if method == "simplex":
        pass

    if method == "dual-simplex":
        pass

    if method == "gomori-pi":
        pass

    if method == "pep":
        pass

    if method == "balas":
        pass

def set_options(usr_options:dict):
    options = {"slacks":0, "display":False}
    for opt in usr_options:
        try:
            options[opt]
            options[opt] = usr_options[opt]
        except KeyError:
            av_opt = "{" + ", ".join(opt for opt in options) + "}" 
            raise Exception(
                f"There is no option {opt}. Available options are:\n{av_opt}"
            )
    return options