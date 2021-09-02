"""
Tools for calculating hydrophobic potential of mean force component in the energy function:

Lin, Matthew S., Nicolas Lux Fawzi, and Teresa Head-Gordon. "Hydrophobic potential of mean force as a solvation function for protein structure prediction." Structure 15.6 (2007): 727-740.

Coded by Jie Li
Jun 4, 2021
"""

from mcsce.core.definitions import aa_atom_type_mappings
from . import libsasa
import numpy as np

##############################
## Parameters in the model
##############################
C1, C2, C3 = [3.81679, 5.46692, 7.11677]
W1, W2, W3= [1.68589, 1.39064, 1.57417]
H1, H2, H3 = [-0.73080, 0.20016, -0.09055]
AC = 6.0

def convert_atom_types(atom_labels, residue_numbers, residue_labels):
    """
    Function for creating a mask that selects atoms that are not carbon-connected hydrogens, prepare the atom types as defined in the SASA calculation.

    Paramters
    ---------
    atom_labels : iterable, list or np.ndarray
        The protein atom labels. Ex: ['N', 'CA, 'C', 'O', 'CB', ...]


    residue_numbers : iterable, list or np.ndarray
        The protein residue numbers per atom in `atom_labels`.
        Ex: [1, 1, 1, 1, 1, 2, 2, 2, 2, ...]

    residue_labels : iterable, list or np.ndarray
        The protein residue labels per atom in `atom_labels`.
        Ex: ['Met', 'Met', 'Met', ...]

    Returns
    ---------
    atom_filter : boolean array for selecting atoms that have SASA atom type definitions from all provided atoms

    new_atom_types: list of the new atom types defined as SASA atom types in their original order, excluding all atoms that do not have type definitions
    """
    atom_filter = []
    new_atom_types = []
    for label, resnum, resname in zip(atom_labels, residue_numbers, residue_labels):
        if resnum == 1:
            if label == "N":
                atom_filter.append(True)
                new_atom_types.append("N-SP3")
                continue
            elif label in ["H1", "H2", "H3"]:
                atom_filter.append(True)
                new_atom_types.append("H-NH+")
                continue
        if label == "OXT":
            atom_filter.append(True)
            new_atom_types.append("O-")
            continue
        new_type = aa_atom_type_mappings[resname][label]
        if new_type is not None:
            atom_filter.append(True)
            new_atom_types.append(new_type)
        else:
            atom_filter.append(False)
    return np.array(atom_filter), new_atom_types

def init_hpmf_calculator(sasa_atom_types, atom_filter, bond_connectivities, threshold=AC):
    """
    Calculate hydrophobic potential of mean force

    Parameters
    ----------
    sasa_atom_types : list of atom types defined in the surface accessible surface area definition

    atom_filter : boolean array for filtering the atoms that have SASA atom type definitions from all atoms provided. The number of True's in this array should be equal to the length of sasa_atom_types

    bond_connectivities : 2d connectivity matrix with shape NxN for all atoms that have SASA atom type definitions, where N is the number of elements in the sasa_atom_types

    threshold : the minimum SA required for a carbon atom to be taken into the summation

    Returns
    ----------
    function closure that takes the 3d coordinates of all atoms (including atoms without sasa atom type definitions) as input and calculates the V_hpmf term in the energy function
    """

    sasa_calculator = libsasa.calc_sasa(sasa_atom_types, bond_connectivities)
    carbon_filter = np.array([name.startswith("C") for name in sasa_atom_types])

    def calculate(coords):
        # calculate solvent accessible surface area
        sasa = sasa_calculator(coords[atom_filter])
        carbon_sasa = sasa[carbon_filter]
        carbon_coords = coords[atom_filter][carbon_filter]
        # create filter for SA_i > A_C
        threshold_carbon = carbon_sasa > threshold
        # calculate pairwise distances for carbons within threshold
        rij = np.linalg.norm(carbon_coords[threshold_carbon, None, :] - carbon_coords[None, threshold_carbon, :], axis=-1)
        # calculate the sum of three gaussians
        gaussians = H1 * np.exp(-((rij - C1) / W1) ** 2) \
                  + H2 * np.exp(-((rij - C2) / W2) ** 2) \
                  + H3 * np.exp(-((rij - C3) / W3) ** 2)
        prefactors = np.tanh(carbon_sasa[threshold_carbon, None]) * np.tanh(carbon_sasa[None, threshold_carbon])
        # masking out the diagonal terms (to exclude j=i terms)
        summing_terms = gaussians * prefactors * (1 - np.eye(np.sum(threshold_carbon)))
        V_hpmf = 0.5 * np.sum(summing_terms)
        return V_hpmf

    return calculate

