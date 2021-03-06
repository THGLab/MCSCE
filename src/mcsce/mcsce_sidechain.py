"""
This file provides a bridge function to allow IDPConformerGenerator to use MCSCE as its side chain builder 

Coded by Jie Li
Aug 31, 2021
"""

from functools import partial

import numpy as np
from mcsce.core import build_definitions
from mcsce.libs.libenergy import prepare_energy_function
from mcsce.libs.libstructure import Structure
from mcsce.core.side_chain_builder import create_side_chain

def mcsce_sidechain(input_seq, coords, n_trials=200, efunc_terms=["lj", "clash"], temperature=300, parallel_worker=16, mode="simple"):
    """
    This function takes an input FASTA string indicating the amino acid sequence, 
    together with an Nx3 array for all coordinates of backbone atoms, 
    and returns an array with shape Mx3 where M is the total number of atoms in the structure with side chains added. 
    This function executes the MCSCE algorithm and returns the lowest energy conformation.

    Parameters
    ----------
    input_seq: str
        A FASTA string to specify the amino acid sequence

    coords: np.array with shape (4L, 3)
        L is the total length of the sequence, and the coordinates are in the order of N, CA, C, O

    n_trials: int
        The total number of trials for the generation procedure

    efunc_terms: list
        Terms to be used in the energy evaluation function

    temperature: float
        The temperature value used for Boltzmann weighting

    parallel_worker: int
        Number of workers for parallel execution

    mode: either simple or exhaustive
        Simple means generating sidechains sequentially and return the first structure without clashes
        Exhaustive means generating n_trials structures and return the lowest energy one

    Returns
    ----------
    full_conformation: np.array with shape (M, 3)
        Coordinates of all atoms in the lowest-energy conformation generated by the MCSCE algorithm. When it is None, it means all trials of conformation generation have failed
    """
    s = Structure(fasta=input_seq)
    s.build()
    s.coords = coords

    ff = build_definitions.forcefields["Amberff14SB"]
    ff_obj = ff(add_OXT=True, add_Nterminal_H=True)

    if mode == "simple":
        return_first_valid = True
    elif mode == "exhaustive":
        return_first_valid = False
    else:
        raise RuntimeError("Mode has to be either simple or exhaustive.")

    final_structure = create_side_chain(s, n_trials, partial(prepare_energy_function, forcefield=ff_obj, terms=efunc_terms), temperature=temperature, parallel_worker=parallel_worker, return_first_valid=return_first_valid)

    if final_structure is not None:
        # final_structure.write_PDB("final.pdb")
        return final_structure.coords
    else:
        return None


# Tests
if __name__ == "__main__":
    fasta = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    backbone_coords = np.array([[27.34 , 24.43 ,  2.614],
       [26.266, 25.413,  2.842],
       [26.913, 26.639,  3.531],
       [27.886, 26.463,  4.263],
       [26.335, 27.77 ,  3.258],
       [26.85 , 29.021,  3.898],
       [26.1  , 29.253,  5.202],
       [24.865, 29.024,  5.33 ],
       [26.849, 29.656,  6.217],
       [26.235, 30.058,  7.497],
       [26.882, 31.428,  7.862],
       [27.906, 31.711,  7.264],
       [26.214, 32.097,  8.771],
       [26.772, 33.436,  9.197],
       [27.151, 33.362, 10.65 ],
       [26.35 , 32.778, 11.395],
       [28.26 , 33.943, 11.096],
       [28.605, 33.965, 12.503],
       [28.638, 35.461, 12.9  ],
       [29.522, 36.103, 12.32 ],
       [27.751, 35.867, 13.74 ],
       [27.691, 37.315, 14.143],
       [28.469, 37.475, 15.42 ],
       [28.213, 36.753, 16.411],
       [29.426, 38.43 , 15.446],
       [30.225, 38.643, 16.662],
       [29.664, 39.839, 17.434],
       [28.85 , 40.565, 16.859],
       [30.132, 40.069, 18.642],
       [29.607, 41.18 , 19.467],
       [30.075, 42.538, 18.984],
       [29.586, 43.57 , 19.483],
       [30.991, 42.571, 17.998],
       [31.422, 43.94 , 17.553],
       [30.755, 44.351, 16.277],
       [31.207, 45.268, 15.566],
       [29.721, 43.673, 15.885],
       [28.978, 43.96 , 14.678],
       [29.604, 43.507, 13.393],
       [29.219, 43.981, 12.301],
       [30.563, 42.623, 13.495],
       [31.191, 42.012, 12.331],
       [30.459, 40.666, 12.13 ],
       [30.253, 39.991, 13.133],
       [30.163, 40.338, 10.886],
       [29.542, 39.02 , 10.653],
       [30.494, 38.261,  9.729],
       [30.849, 38.85 ,  8.706],
       [30.795, 37.015, 10.095],
       [31.72 , 36.289,  9.176],
       [30.955, 35.211,  8.459],
       [30.025, 34.618,  9.04 ],
       [31.244, 34.986,  7.197],
       [30.505, 33.884,  6.512],
       [31.409, 32.68 ,  6.446],
       [32.619, 32.812,  6.125],
       [30.884, 31.485,  6.666],
       [31.677, 30.275,  6.639],
       [31.022, 29.288,  5.665],
       [29.809, 29.395,  5.545],
       [31.834, 28.412,  5.125],
       [31.22 , 27.341,  4.275],
       [31.44 , 26.079,  5.08 ],
       [32.576, 25.802,  5.461],
       [30.31 , 25.458,  5.384],
       [30.288, 24.245,  6.193],
       [29.279, 23.227,  5.641],
       [28.478, 23.522,  4.725],
       [29.38 , 22.057,  6.232],
       [28.468, 20.94 ,  5.98 ],
       [27.819, 20.609,  7.316],
       [28.449, 20.674,  8.36 ],
       [26.559, 20.22 ,  7.288],
       [25.829, 19.825,  8.494],
       [26.541, 18.732,  9.251],
       [26.333, 18.536, 10.457],
       [27.361, 17.959,  8.559],
       [28.054, 16.835,  9.21 ],
       [29.258, 17.318,  9.984],
       [29.93 , 16.477, 10.606],
       [29.599, 18.599,  9.828],
       [30.796, 19.083, 10.566],
       [30.491, 19.162, 12.04 ],
       [29.367, 19.523, 12.441],
       [31.51 , 18.936, 12.852],
       [31.398, 19.064, 14.286],
       [31.593, 20.553, 14.655],
       [32.159, 21.311, 13.861],
       [31.113, 20.863, 15.86 ],
       [31.288, 22.201, 16.417],
       [32.776, 22.519, 16.577],
       [33.233, 23.659, 16.384],
       [33.548, 21.526, 16.95 ],
       [35.031, 21.722, 17.069],
       [35.615, 22.19 , 15.759],
       [36.532, 23.046, 15.724],
       [35.139, 21.624, 14.662],
       [35.59 , 21.945, 13.302],
       [35.238, 23.382, 12.92 ],
       [36.066, 24.109, 12.333],
       [34.007, 23.745, 13.25 ],
       [33.533, 25.097, 12.978],
       [34.441, 26.099, 13.684],
       [34.883, 27.09 , 13.093],
       [34.734, 25.822, 14.949],
       [35.596, 26.715, 15.736],
       [36.975, 26.826, 15.107],
       [37.579, 27.926, 15.159],
       [37.499, 25.743, 14.571],
       [38.794, 25.761, 13.88 ],
       [38.728, 26.591, 12.611],
       [39.704, 27.346, 12.277],
       [37.633, 26.543, 11.867],
       [37.471, 27.391, 10.668],
       [37.441, 28.882, 11.052],
       [38.02 , 29.772, 10.382],
       [36.811, 29.17 , 12.192],
       [36.731, 30.57 , 12.645],
       [38.148, 30.981, 13.069],
       [38.544, 32.15 , 12.856],
       [38.883, 30.11 , 13.713],
       [40.269, 30.508, 14.115],
       [41.092, 30.808, 12.851],
       [41.828, 31.808, 12.681],
       [41.001, 29.878, 11.931],
       [41.718, 30.022, 10.643],
       [41.399, 31.338,  9.967],
       [42.26 , 32.036,  9.381],
       [40.117, 31.75 ,  9.988],
       [39.808, 32.994,  9.233],
       [39.837, 34.271,  9.995],
       [40.164, 35.323,  9.345],
       [39.655, 34.335, 11.285],
       [39.676, 35.547, 12.072],
       [40.675, 35.527, 13.2  ],
       [40.814, 36.528, 13.911],
       [41.317, 34.393, 13.432],
       [42.345, 34.269, 14.431],
       [41.949, 34.076, 15.842],
       [42.829, 34.   , 16.739],
       [40.642, 33.916, 16.112],
       [40.226, 33.716, 17.509],
       [40.449, 32.278, 17.945],
       [39.936, 31.336, 17.315],
       [41.189, 32.085, 19.031],
       [41.461, 30.751, 19.594],
       [40.168, 30.026, 19.918],
       [39.264, 30.662, 20.521],
       [40.059, 28.758, 19.607],
       [38.817, 28.02 , 19.889],
       [38.421, 28.048, 21.341],
       [37.213, 28.036, 21.704],
       [39.374, 28.09 , 22.24 ],
       [39.063, 28.063, 23.695],
       [38.365, 29.335, 24.159],
       [37.684, 29.39 , 25.221],
       [38.419, 30.373, 23.341],
       [37.738, 31.637, 23.712],
       [36.334, 31.742, 23.087],
       [35.574, 32.618, 23.483],
       [36.   , 30.86 , 22.172],
       [34.738, 30.875, 21.473],
       [33.589, 30.189, 22.181],
       [33.58 , 29.009, 22.499],
       [32.478, 30.917, 22.269],
       [31.2  , 30.329, 22.78 ],
       [30.21 , 30.509, 21.65 ],
       [29.978, 31.726, 21.269],
       [29.694, 29.436, 21.054],
       [28.762, 29.573, 19.906],
       [27.331, 29.317, 20.364],
       [27.101, 28.346, 21.097],
       [26.436, 30.232, 20.004],
       [25.034, 30.17 , 20.401],
       [24.101, 30.149, 19.196],
       [24.196, 30.948, 18.287],
       [23.141, 29.187, 19.241],
       [22.126, 29.062, 18.183],
       [20.835, 28.629, 18.904],
       [20.821, 27.734, 19.749],
       [19.81 , 29.378, 18.578],
       [18.443, 29.143, 19.083],
       [18.453, 28.941, 20.591],
       [17.86 , 27.994, 21.128],
       [19.172, 29.808, 21.243],
       [19.399, 29.894, 22.655],
       [20.083, 28.729, 23.321],
       [19.991, 28.584, 24.561],
       [20.801, 27.931, 22.578],
       [21.55 , 26.796, 23.133],
       [23.046, 27.087, 22.913],
       [23.383, 27.627, 21.87 ],
       [23.88 , 26.727, 23.851],
       [25.349, 26.872, 23.643],
       [25.743, 25.586, 22.922],
       [25.325, 24.489, 23.378],
       [26.465, 25.689, 21.833],
       [26.826, 24.521, 21.012],
       [27.994, 23.781, 21.643],
       [28.904, 24.444, 22.098],
       [27.942, 22.448, 21.648],
       [29.015, 21.657, 22.288],
       [29.942, 21.106, 21.24 ],
       [29.47 , 20.677, 20.19 ],
       [31.233, 21.09 , 21.459],
       [32.262, 20.67 , 20.514],
       [32.128, 19.364, 19.75 ],
       [32.546, 19.317, 18.558],
       [31.697, 18.311, 20.406],
       [31.568, 16.962, 19.825],
       [30.32 , 16.698, 19.051],
       [30.198, 15.657, 18.366],
       [29.34 , 17.594, 19.076],
       [28.108, 17.439, 18.276],
       [28.375, 17.999, 16.887],
       [29.326, 18.786, 16.69 ],
       [27.51 , 17.689, 15.954],
       [27.574, 18.192, 14.563],
       [26.482, 19.28 , 14.432],
       [25.609, 19.388, 15.287],
       [26.585, 20.063, 13.378],
       [25.594, 21.109, 13.072],
       [24.241, 20.436, 12.857],
       [23.264, 20.951, 13.329],
       [24.24 , 19.233, 12.246],
       [22.924, 18.583, 12.025],
       [22.229, 18.244, 13.325],
       [20.963, 18.253, 13.395],
       [22.997, 17.978, 14.366],
       [22.418, 17.638, 15.693],
       [21.46 , 18.737, 16.163],
       [20.497, 18.506, 16.9  ],
       [21.846, 19.954, 15.905],
       [21.079, 21.149, 16.251],
       [20.142, 21.59 , 15.149],
       [19.499, 22.645, 15.321],
       [19.993, 20.884, 14.049],
       [19.065, 21.352, 12.999],
       [19.442, 22.745, 12.51 ],
       [18.571, 23.61 , 12.289],
       [20.717, 22.964, 12.26 ],
       [21.184, 24.263, 11.69 ],
       [21.11 , 24.111, 10.173],
       [21.841, 23.198,  9.686],
       [20.291, 24.875,  9.507],
       [20.081, 24.773,  8.033],
       [20.822, 25.914,  7.332],
       [21.323, 26.83 ,  8.008],
       [20.924, 25.862,  6.006],
       [21.656, 26.847,  5.24 ],
       [21.127, 28.24 ,  5.574],
       [19.958, 28.465,  5.842],
       [22.099, 29.163,  5.605],
       [21.907, 30.563,  5.881],
       [21.466, 30.953,  7.261],
       [21.066, 32.112,  7.533],
       [21.674, 30.034,  8.191],
       [21.419, 30.253,  9.62 ],
       [22.504, 31.228, 10.136],
       [23.579, 31.321,  9.554],
       [22.241, 31.873, 11.241],
       [23.212, 32.762, 11.891],
       [23.509, 32.224, 13.29 ],
       [22.544, 31.942, 14.034],
       [24.79 , 32.021, 13.618],
       [25.149, 31.609, 14.98 ],
       [25.698, 32.876, 15.669],
       [26.158, 33.73 , 14.894],
       [25.621, 32.945, 16.95 ],
       [26.179, 34.127, 17.65 ],
       [27.475, 33.651, 18.304],
       [27.507, 32.587, 18.958],
       [28.525, 34.447, 18.189],
       [29.801, 34.145, 18.829],
       [30.052, 35.042, 20.004],
       [30.105, 36.305, 19.788],
       [30.124, 34.533, 21.191],
       [30.479, 35.369, 22.374],
       [31.901, 34.91 , 22.728],
       [32.19 , 33.696, 22.635],
       [32.763, 35.831, 23.09 ],
       [34.145, 35.472, 23.481],
       [34.239, 35.353, 24.979],
       [33.707, 36.197, 25.728],
       [34.93 , 34.384, 25.451],
       [35.161, 34.174, 26.896],
       [36.671, 34.296, 27.089],
       [37.305, 33.233, 26.795],
       [37.197, 35.397, 27.513],
       [38.668, 35.502, 27.68 ],
       [39.076, 34.931, 29.031],
       [38.297, 34.946, 29.996],
       [40.294, 34.412, 29.045],
       [40.873, 33.802, 30.253],
       [41.765, 34.829, 30.944],
       [42.945, 34.994, 30.583],
       [41.165, 35.531, 31.898],
       [41.845, 36.55 , 32.686],
       [41.251, 37.941, 32.588],
       [41.102, 38.523, 31.5  ],
       [40.946, 38.472, 33.757],
       [40.373, 39.813, 33.944],
       [40.031, 39.992, 35.432],
       [38.933, 40.525, 35.687]])
    result = mcsce_sidechain(fasta, backbone_coords, n_trials=25, efunc_terms=["lj", "clash"])
    print(result)