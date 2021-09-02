"""
Tools for calculating solvent accessible surface area following:

Winnfried Hasel, Thomas F. Hendrickson, W.Clark Still,
A rapid approximation to the solvent accessible surface areas of atoms,
Tetrahedron Computer Methodology,
Volume 1, Issue 2,
1988,
Pages 103-116,
ISSN 0898-5529,
https://doi.org/10.1016/0898-5529(88)90015-2.

Coded by Jie Li
Jun 3, 2021
"""

import numpy as np
from mcsce.libs.libcalc import *
from numba import njit

P_PARAMS = {
    "C-SP3": 2.149,
    "CH-SP3": 1.276,
    "CH2-SP3": 1.045,
    "CH3-SP3": 0.880,
    "C-SP2": 1.554,
    "CH-SP2": 1.073,
    "CH2-SP2": 0.961,
    "C-SP": 0.737,
    "H-OH": 0.944,
    "H-NH": 1.128,
    "H-NH+": 1.049,
    "H-SH": 0.928,
    "O-SP3": 1.080,
    "O-SP2": 0.926,
    "O-": 0.922,
    "N-SP3": 1.215,
    "N-SP2": 1.413,
    "NH,=N": 1.028,
    "S": 1.121,
    "F": 0.906,
    "Cl": 0.906,
    "Br": 0.898,
    "I": 0.876
}

R_PARAMS = {
    "C-SP3": 1.70,
    "CH-SP3": 1.80,
    "CH2-SP3": 1.90,
    "CH3-SP3": 2.00,
    "C-SP2": 1.72,
    "CH-SP2": 1.80,
    "CH2-SP2": 1.80,
    "C-SP": 1.78,
    "H-OH": 1.00,
    "H-NH": 1.10,
    "H-NH+": 1.20,
    "H-SH": 1.20,
    "O-SP3": 1.52,
    "O-SP2": 1.50,
    "O-": 1.70,
    "N-SP3": 1.60,
    "N-SP2": 1.55,
    "NH,=N": 1.55,
    "S": 1.80,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98
}

PIJ_BONDED = 0.8875
PIJ_NON_BONDED = 0.3516

def calc_sasa(atom_types, bond_connectivities, r_solvent=1.4):
    """
    Main calculator for solvent accessible surface area: anything that is independent of the coordinate positions will be calculated here and be directly used to enhance calculation speed

    Params:
    ----------
        atom_types: list (array) of atom type strings
        bond_connectivities: NxN boolean matrix indicating the bond connectivities of all atoms (np.array)
        r_solvent: radius of solvent probe in the unit of angstrom (float)
    """

    Rs = r_solvent
    Ri = np.array([R_PARAMS[a] for a in atom_types])
    Pi = np.array([P_PARAMS[a] for a in atom_types])
    Pij = bond_connectivities * PIJ_BONDED + (1 - bond_connectivities) * PIJ_NON_BONDED
    Si = 4 * np.pi * (Ri + r_solvent) ** 2
    summing_thresholds = Ri[:, None] + Ri[None, :] + 2 * Rs
    N = len(atom_types)


    def calc(coords):
        """
        Main function for calculating solvent accessible surface area from the given coordinates
        """
        dij = np.linalg.norm(coords.reshape(N, 1, 3) - coords.reshape(1, N, 3), axis=-1)
        mask = dij < summing_thresholds
        mask *= (1 - np.eye(N)).astype(bool)
        bij = np.pi * (Ri + Rs).reshape(1, -1) * (summing_thresholds - dij) * \
             (1 + (Ri.reshape(1, -1) - Ri.reshape(-1, 1)) / (dij + 1e-8))
        product_terms = 1 - Pi.reshape(-1, 1) * Pij * bij / Si.reshape(-1, 1)
        product_terms[~mask] = 1
        Ai = Si * np.prod(product_terms, axis=-1)
        return Ai
    return calc