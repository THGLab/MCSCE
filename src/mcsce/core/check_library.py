"""
This file defines the function that reads the Dunbrack protein side chain rotamer library

Shapovalov, Maxim V., and Roland L. Dunbrack Jr. "A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions." Structure 19.6 (2011): 844-858.

Coded by Jie Li
Date created: Jul 28, 2021
Modified by Oufan Zhang to add ptm rotamers
"""

from pathlib import Path
import numpy as np
from mcsce.core.definitions import ptm_aa, ptm_h

_filepath = Path(__file__).resolve().parent  # folder
_library_path = _filepath.joinpath('data', 'SimpleOpt1-5', 'ALL.bbdep.rotamers.lib')
_ptm_library_path = _filepath.joinpath('data', 'ptm_rotamer.lib')


def get_closest_angle(input_angle):
    if 0 < input_angle < 120:
        return 1
    elif -120 < input_angle < 0:
        return 2
    else:
        return 3

def calc_total_prob(residue):
    data = {1: {"counts": [], "probs": []},
            2: {"counts": [], "probs": []},
            3: {"counts": [], "probs": []},
           }
    with open(_library_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(residue):
                restype, phi, psi, count, _, _, _, _, prob, chi1, chi2, chi3, chi4, _, _, _, _= line.split()
                assert restype == residue
                chi1_range = get_closest_angle(float(chi1))
                data[chi1_range]["counts"].append(float(count)) 
                data[chi1_range]["probs"].append(float(prob))

    for n in range(1, 4):
        print(np.sum(np.array(data[n]["counts"]) * np.array(data[n]["probs"]) / np.sum(data[n]["counts"])))    

if __name__ == "__main__":
    for resn in ["SER", "THR", "TYR"]:
        print(resn) 
        calc_total_prob(resn)
