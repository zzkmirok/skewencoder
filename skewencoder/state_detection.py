__all__ = ["Bond_type_lib",
           "transform_colvar_key",
           "parse_unbiased_colvar",
           "State_detection"]
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from skewencoder.switchfunction import SwitchFun 
import numpy as np
from collections.abc import Sequence, Mapping, Set
from typing import Tuple, Union
from skewencoder.io import load_dataframe

from typing import Callable

from functools import partial

import re

# TODO: should be static

class Bond_type_lib:
    def __init__(self):
        self.bond_type_dict : Mapping[str, Mapping[str, Union[float, Tuple[int, int]]]] = {}

    # TODO: first just for KHP, also could be static?
    # TODO: Not only update m, n, but also possible bonds (heavy atom pairs) that was not read from unbiased colvar
    def build_default(self):
        self.bond_type_dict["C-C"] = {"m,n" : (12, 6), "bond length": 1.5}
        self.bond_type_dict["C-O"] = {"m,n" : (12, 6), "bond length": 1.5}
        self.bond_type_dict["O-O"] = {"m,n" : (12, 6), "bond length": 1.5}
        self.bond_type_dict["C-N"] = {"m,n" : (12, 6), "bond length": 1.5}
        self.bond_type_dict["N-O"] = {"m,n" : (12, 6), "bond length": 1.5}
        self.bond_type_dict["O-H"] = {"m,n" : (12, 6), "bond length": 1.1} # including OH bond
        self.bond_type_dict["H-C"] = {"m,n" : (12, 6), "bond length": 1.1} # including CH bond
    
    def _update_m_n(self):
        self.bond_type_dict["c-c"] = {"m,n" : (12, 6)}
        self.bond_type_dict["c-o"] = {"m,n" : (12, 6)}
        self.bond_type_dict["o-h"] = {"m,n" : (12, 6)} # including OH bond
        self.bond_type_dict["c-h"] = {"m,n" : (12, 6)} # including CH bond
    
    def __call__(self, bond_type_dict):
        self._update_m_n()
        for key in bond_type_dict:
            if key in self.bond_type_dict:
                bond_type_dict[key].update(self.bond_type_dict[key])
        
        return bond_type_dict

def transform_colvar_key(colvar_key: str, pattern: str = r"^([A-Za-z]+)\d+([A-Za-z]+)\d+$") -> str:
    match = re.match(pattern, colvar_key)
    if match:
        # TODO: check this maybe better to first save as Set then str because Set has a internal order?
        atom_pair = {match.group(1), match.group(2)} 
        bond_type = None
        if len(atom_pair) == 1:
            bond_type = f"{list(atom_pair)[0]}-{list(atom_pair)[0]}"
        else:
            bond_type = f"{sorted(list(atom_pair))[0]}-{sorted(list(atom_pair))[1]}"
        return bond_type
    else:
        return None



def parse_unbiased_colvar(colvar_file: str = f"{SCRIPT_DIR}/COLVAR", std_tol: float = 0.05, r0_tol: float = 1.6, transform_colvar_key : Callable[[str],str] = partial(transform_colvar_key, pattern = r"^([A-Za-z]+)\d+([A-Za-z]+)\d+$")):
    colvar_df = load_dataframe(colvar_file)
    last_rows = colvar_df["time"].values.shape[0]
    bond_type_dict : Mapping[str, Mapping[str, Union[float, Tuple[int, int]]]] = {} #TODO: should be bond type dict?
    heavy_atom_pairs_list: Sequence[str] = []
    # pattern = r"^([A-Za-z]+)\d+([A-Za-z]+)\d+$" # Pattern depends on the keys in COLVAR file
    n_heavy_atom_pairs = 0
    for key in colvar_df.keys():
        current_bond_type = transform_colvar_key(colvar_key=key)
        if current_bond_type is not None:
            n_heavy_atom_pairs += 1
            heavy_atom_pairs_list.append(key)
            current_std = np.std(colvar_df[key].values)
            current_r0 = np.mean(colvar_df[key].iloc[-last_rows:].values)
            if current_bond_type not in bond_type_dict.keys():
                if current_r0 < r0_tol:
                    bond_type_dict[current_bond_type] = {"bond length": current_r0}
                else:
                    bond_type_dict[current_bond_type] = {"bond length": r0_tol}
            else:
                if current_std <= std_tol and current_r0 < r0_tol:
                    temp_r0 = np.min((bond_type_dict[current_bond_type]["bond length"], r0_tol))
                    if temp_r0 == r0_tol:
                        bond_type_dict[current_bond_type].update({"bond length": current_r0})
                    else:
                        if current_r0 > temp_r0:
                            bond_type_dict[current_bond_type].update({"bond length": current_r0})
                        
    
    return Bond_type_lib()(bond_type_dict=bond_type_dict), n_heavy_atom_pairs, heavy_atom_pairs_list
     

# TODO: at @properties functions
class State_detection:
    def __init__(self, interval: Tuple[float, float], bond_type_dict: 
                 Mapping[str, Mapping[str, Union[float, Tuple[int, int]]]], n_heavy_atom_pairs: int, pattern : str = r"^([A-Za-z]+)\d+([A-Za-z]+)\d+$"):
        self.interval = interval
        self.bond_type_dict = bond_type_dict
        self.sw_dict: Mapping[str, SwitchFun] = {}

        self.states: int = [0] # TODO: may be static
        self.states_connectivity: Sequence[np.ndarray] = [] # TODO: Also sparse? So far 1-D array only heavy atoms
        self.n_heay_atom_pairs = n_heavy_atom_pairs
        self.current_state = 0
        self.pattern = pattern
        self._transform_colvar_key = partial(transform_colvar_key, pattern = self.pattern)
        for bond_name, options in self.bond_type_dict.items():
            m = 12
            n = 6
            bond_length_boundary = None
            for opt, opt_value in options.items():
                 # TODO: A random number for init the variable, might change later
                # Might change to switch case
                if opt == "m,n":
                    if not (isinstance(opt_value, tuple) and all(isinstance(i, int) for i in opt_value)):
                        raise TypeError(f"Wrong type for option {opt} {type(opt_value)}")
                    else:
                        m = opt_value[0]
                        n = opt_value[1]
                elif opt == "bond length":
                    if not isinstance(opt_value, float):
                        raise TypeError(f"Wrong type for option {opt} {type(opt_value)}")
                    else:
                        bond_length_boundary = opt_value * 1.2
                else:
                    raise ValueError()
            # TODO: More exception handling
            if bond_length_boundary is None:
                raise ValueError()
            else:
                temp_switch_fun = SwitchFun(bond_length_boundary, options={"m": m, "n": n})
            self.sw_dict[bond_name] = temp_switch_fun
    
    def __call__(self, colvar_file: str): # TODO: directly colvar file?
        colvar_df = load_dataframe(colvar_file)
        last_rows = colvar_df["time"].values.shape[0]//4
        is_stable_state = True
        current_state_connectivity = np.zeros(self.n_heay_atom_pairs, dtype=bool)
        # TODO: discover if I update the values via colvar_df.update({key: self.sw_dict[current_bond_type](value)}), my value will then be updated?
        iter_key = 0
        for key in colvar_df.keys():
            print(f"current key is {key}")
            current_bond_type = self._transform_colvar_key(colvar_key=key)
            if current_bond_type is not None:
                last_rows_mean = np.mean(colvar_df[key].iloc[-last_rows:].values)
                print(f"current bond type: {current_bond_type}")
                last_rows_mean = self.sw_dict[current_bond_type](last_rows_mean)
                print(f"iter = {iter_key}, last_rows_mean={last_rows_mean}, key = {key}")
                if last_rows_mean >= self.interval[0] and last_rows_mean <= self.interval[1]:
                    print(f"Not Stable atoms pair:\n iter = {iter_key}, last_rows_mean={last_rows_mean}, key = {key}")
                    self.current_state = 0
                    is_stable_state = False
                    return is_stable_state, False
                elif last_rows_mean > self.interval[1]:
                    current_state_connectivity[iter_key] = 1
                else:
                    current_state_connectivity[iter_key] = 0
                iter_key += 1
        
        if 1 not in current_state_connectivity:
            raise ValueError("All bonds btw heavy atoms broken, simulation stop")

        is_new_state = True

        if len(self.states_connectivity) == 0:
            self.states_connectivity.append(current_state_connectivity)
        else:
            for i, state_connectivity in enumerate(self.states_connectivity):
                if (state_connectivity == current_state_connectivity).all():
                    is_new_state = False
                    self.current_state = self.states[i + 1]
                    break
            if is_new_state:
                self.states_connectivity.append(current_state_connectivity)

        if is_new_state:
            self.states.append(self.states[-1] + 1)
            self.current_state = self.states[-1]

        return is_stable_state, is_new_state

    @property
    def n_states(self):
        return len(self.states)-1
    

def test_State_detection():
    x = np.array([[1.1, 1.8, 1.2, 1.3], [1.2, 2.0, 1.1, 1.5]])
    '''
    bond_type_dict = {
                    "c-c": {"m,n" : (8, 6), "bond length": 1.5}, 
                    "o-o": {"bond length": 1.4}
                    }
    '''
    bond_type_dict, n_heavy_atom_pairs, heavy_atom_pairs = parse_unbiased_colvar()
    print(bond_type_dict)
    test_state_detection = State_detection((0.3, 0.7), bond_type_dict=bond_type_dict, n_heavy_atom_pairs=n_heavy_atom_pairs)
    for key, value in test_state_detection.sw_dict.items():
        print(f"{key}:{value}\n")
        if key == "c-c":
           x[:, :2] = value(x[:, :2])
        elif key == "o-o":
            x[:, 2:3] = value(x[:, 2:3])
        else:
            x[:, 3:4] = value(x[:, 3:4])

    print(test_state_detection(f"{SCRIPT_DIR}/COLVAR"))
    print(test_state_detection.states)
    print(test_state_detection.states_connectivity)
    print(test_state_detection.current_state)

def test_parse_unbiased_colvar():
    parse_unbiased_colvar()

if __name__ == "__main__":
    test_State_detection()
    # test_parse_unbiased_colvar()
        