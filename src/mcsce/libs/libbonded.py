"""
Code for calculating bonded energy terms, including angle and dihedral terms from the forcefield

Coded by Jie Li
Jul 2, 2021
"""

import numpy as np
from mcsce.libs.libparse import extract_ff_params_for_seq
from mcsce.libs.libcalc import calc_angle_coords, calc_torsion_angles
from itertools import combinations


def prepare_angles_and_dihedrals(angle_head_atom_matrix, dihedral_head_atom_matrix, connectivity_matrix, atom_labels, residue_numbers, residue_labels, forcefield):
    '''
    Function for providing all angles in the protein

    params
    ----------
    angle_head_atom_matrix: np.ndarray or None
        an NxN matrix with upper triangle representing all atom pairs separated by 2 bonds (head atoms of an angle)
        when is None, do not prepare angle terms calculation

    dihedral_head_atom_matrix: np.ndarray or None
        an NxN matrix with upper triangle representing all atom pairs separated by 3 bonds (head atoms of a dihedral)
        when is None, do not prepare dihedral terms calculation

    connectivity_matrix: np.ndarray
        an NxN matrix representing the connectivity of all atoms in the protein

    atom_labels : iterable, list or np.ndarray
        The protein atom labels. Ex: ['N', 'CA, 'C', 'O', 'CB', ...]

    residue_numbers : iterable, list or np.ndarray
        The protein residue numbers per atom in `atom_labels`.
        Ex: [1, 1, 1, 1, 1, 2, 2, 2, 2, ...]

    residue_labels : iterable, list or np.ndarray
        The protein residue labels per atom in `atom_labels`.
        Ex: ['Met', 'Met', 'Met', ...]
    '''
    prepare_angles = angle_head_atom_matrix is not None
    prepare_dihedrals = dihedral_head_atom_matrix is not None
    angle_involved_atoms = []
    dihedral_involved_atoms = []
    index_arr = np.arange(len(atom_labels))
    for atom1_idx in range(len(atom_labels)):
        for atom2_idx in range(atom1_idx + 1, len(atom_labels)):
            if prepare_angles:
                if angle_head_atom_matrix[atom1_idx, atom2_idx]:
                    # atom1 and atom2 are 2 bonds away -> an angle can be defined
                    # atoms in the middle are found by searching atom1 connected atoms and atom2 connected atoms, and find the same index of these two
                    atom1_connected = index_arr[connectivity_matrix[atom1_idx].astype(bool)]
                    atom2_connected = index_arr[connectivity_matrix[atom2_idx].astype(bool)]
                    middle_atom = set(atom1_connected) & set(atom2_connected)

                    assert len(middle_atom) == 1
                    middle_atom = middle_atom.pop()
                    atomic_idx_tuple = (atom1_idx, middle_atom, atom2_idx)
                    angle_involved_atoms.append(atomic_idx_tuple)

            if prepare_dihedrals:
                if dihedral_head_atom_matrix[atom1_idx, atom2_idx]:
                    # atom1 and atom2 are 3 bonds away -> a dihedral can be defined
                    # atoms in the middle are found by searching atom1 connected atoms and atom2 connected atoms, and find the two of them that are within each other's neighbors
                    found = False
                    atom1_connected = index_arr[connectivity_matrix[atom1_idx].astype(bool)]
                    atom2_connected = index_arr[connectivity_matrix[atom2_idx].astype(bool)]
                    for atom1_neighbor in atom1_connected:
                        for atom2_neighbor in atom2_connected:
                            if connectivity_matrix[atom1_neighbor, atom2_neighbor]:
                                # atoms involved in this dihedral have been found
                                atomic_idx_tuple = (atom1_idx, atom1_neighbor, atom2_neighbor, atom2_idx)
                                dihedral_involved_atoms.append(atomic_idx_tuple)
                                found = True
                                break
                        if found:
                            break

    all_atom_types = extract_ff_params_for_seq(
                    atom_labels,
                    residue_numbers,
                    residue_labels,
                    forcefield,
                    'atom_type',
                    )

    
    angle_dict = {}
    dihedral_dict = {}
    improper_dihedral_dict = {}
    if prepare_angles:
        for angle in angle_involved_atoms:
            angle_dict[angle] = extract_matched_angle_definitions(tuple(all_atom_types[atom_idx] for atom_idx in angle), forcefield)
    if prepare_dihedrals:
        for dihedral in dihedral_involved_atoms:
            dihedral_dict[dihedral] = extract_matched_proper_dihedral_definitions(tuple(all_atom_types[atom_idx] for atom_idx in dihedral), forcefield) # Whether there is multiple counting?

        # Prepare improper dihedrals
        improper_dih_ff_terms = extract_all_improper_dihedrals_from_forcefield(forcefield)
        for central_atom_idx in range(len(atom_labels)):
            central_atom_type = all_atom_types[central_atom_idx]
            if central_atom_type in improper_dih_ff_terms:
                connected_idx = index_arr[connectivity_matrix[central_atom_idx].astype(bool)]
                if len(connected_idx) < 3:
                    continue
                connected_atom_types = [all_atom_types[idx] for idx in connected_idx]
                for connected_atoms in improper_dih_ff_terms[central_atom_type]:
                    # consider possibility that multiple improper dihedral may exist for the same central atom
                    connected_idx_duplicated = [item for item in connected_idx]
                    connected_atom_types_duplicated = [item for item in connected_atom_types]
                    idx = []
                    appendix = [(), ]
                    match = True
                    for atom_type in sorted(connected_atoms, reverse=True): # reversed sort so that wildcard are at the end
                        if atom_type == "":
                            # the remaining atoms can all be matches
                            possible_indices = combinations(range(len(connected_idx_duplicated)), 3 - len(idx))
                            appendix = [[connected_idx_duplicated[n] for n in indices] for indices in possible_indices]
                            break
                        if atom_type in connected_atom_types_duplicated:
                            index = connected_atom_types_duplicated.index(atom_type)
                            idx.append(connected_idx_duplicated[index])
                            del connected_idx_duplicated[index]
                            del connected_atom_types_duplicated[index]
                        else:
                            # an atom supposed to be connected to the central atom is not in the residue: this match need not be considered
                            match = False
                            break
                    if match:
                        for a in appendix:
                            improper_dihedral_dict[(central_atom_idx, ) + tuple(idx) + tuple(a)] = improper_dih_ff_terms[central_atom_type][connected_atoms]
        # matching duplicates when encountering wildcards?
    return angle_dict, dihedral_dict, improper_dihedral_dict

def extract_matched_angle_definitions(atom_type_tuple, forcefield):
    '''
    Obtain all angle term definitions from the forcefield with matched atom types, including reversed order

    * There is no general atom type defined for angles
    '''
    angle_terms = []
    for combination in [atom_type_tuple,
                         atom_type_tuple[::-1]]:
        if combination in forcefield:
            angle_terms.append(forcefield[combination])
    return angle_terms

def extract_matched_proper_dihedral_definitions(atom_type_tuple, forcefield):
    '''
    Obtain all proper dihedral term definitions from the forcefield with matched atom types, including those "general" atom types and reverse

    * General atom types will only occur for either both atom1 & atom4, or any/both of atom2 & atom3
    '''
    dihedral_terms = []
    for combination in [atom_type_tuple,
                         atom_type_tuple[::-1],
                         ("", atom_type_tuple[1], atom_type_tuple[2], ""),
                         ("", atom_type_tuple[2], atom_type_tuple[1], ""),
                         (atom_type_tuple[0], "", "", atom_type_tuple[3]),
                         (atom_type_tuple[3], "", "", atom_type_tuple[0]),
                         (atom_type_tuple[0], atom_type_tuple[1], "", atom_type_tuple[3]),
                         (atom_type_tuple[3], "", atom_type_tuple[1], atom_type_tuple[0]),
                         (atom_type_tuple[0], "", atom_type_tuple[2], atom_type_tuple[3]),
                         (atom_type_tuple[3], atom_type_tuple[2], "", atom_type_tuple[0])]:
        if combination in forcefield:
            if forcefield[combination]['tag'] == 'Proper':
                k_s = [k for k in forcefield[combination] if k.startswith('k')] # coefficient k's
                k_values = [float(forcefield[combination][k]) for k in k_s]
                if (np.array(k_values) != 0).any():
                    dihedral_terms.append(forcefield[combination]) # only add dihedral term for k!=0
    return dihedral_terms
    

def extract_all_improper_dihedrals_from_forcefield(forcefield):
    """
    Loop through all improper dihedral terms in the forcefield and organize into the format 
    {central_atom_type: {connected_atom_types: {ks, periodicities, phases}}}
    """
    improper_collection = {}
    for item in forcefield:
        if type(item) is tuple and len(item) == 4:
            # This is a dihedral term definition
            if forcefield[item]['tag'] == 'Improper':
                if float(forcefield[item]['k1']) != 0:
                    central_atom_type = item[0]
                    connected_atom_types = item[1:]
                    if central_atom_type in improper_collection:
                        improper_collection[central_atom_type][connected_atom_types] = forcefield[item]
                    else:
                        improper_collection[central_atom_type] = {connected_atom_types: forcefield[item]}
    return improper_collection

def init_angle_calculator(angles):
    """
    Definition for the angle term calculator: Taken as formula E = k * (actual_angle - ideal_angle) ** 2
    """
    angle_atom_indices = []
    ks = []
    ideal_angles = []
    for item in angles:
        for term in angles[item]:
            angle_atom_indices.append(item)
            ks.append(term['k'])
            ideal_angles.append(term['angle'])
    ks = np.array(ks, dtype=float)
    ideal_angles = np.array(ideal_angles, dtype=float)

    def calc(coords):
        actual_angles = np.array([calc_angle_coords(coords[list(idx)]) for idx in angle_atom_indices])
        energies = ks * (actual_angles - ideal_angles) ** 2
        return np.sum(energies)

    return calc

def init_dihedral_calculator(dihedrals, improper_dihedrals):
    """
    Definition for the dihedral term calculator: Taken as formula E = k * (1 + cos(periocity * dih - phase))

    Proper dihedrals are connected in 1-2-3-4 order
    Improper dihedrals are connected by 2-1-3
                                          |
                                          4
    """
    dihedral_atom_indices = []
    ks = []
    periodicities = []
    phases = []
    dihedral_atom_indices_improper = []
    ks_improper = []
    periodicities_improper = []
    phases_improper = []
    for item in dihedrals:
        for term in dihedrals[item]:
            for kn in ['1', '2', '3', '4']:
                if 'k'+kn in term and float(term['k'+kn]) != 0:
                    dihedral_atom_indices.append(item)
                    ks.append(term['k'+kn])
                    periodicities.append(term['periodicity'+kn])
                    phases.append(term['phase'+kn])
    for item in improper_dihedrals:
        dihedral_atom_indices_improper.append(item)
        ks_improper.append(improper_dihedrals[item]['k1'])
        periodicities_improper.append(improper_dihedrals[item]['periodicity1'])
        phases_improper.append(improper_dihedrals[item]['phase1'])
    ks = np.array(ks, dtype=float)
    periodicities = np.array(periodicities, dtype=float)
    phases = np.array(phases, dtype=float)
    ks_improper = np.array(ks_improper, dtype=float)
    periodicities_improper = np.array(periodicities_improper, dtype=float)
    phases_improper = np.array(phases_improper, dtype=float)

    def calc(coords):
        dihedrals = np.concatenate([calc_torsion_angles(coords[list(idx)]) for idx in dihedral_atom_indices])
        improper_dihedrals = np.concatenate([calc_improper_torsion_angles(coords[list(idx)]) for idx in dihedral_atom_indices])
        energies = ks * (1 + np.cos(periodicities * dihedrals - phases))
        energies_improper = ks_improper * (1 + np.cos(periodicities_improper * improper_dihedrals - phases_improper))
        return np.sum(energies) + np.sum(energies_improper)

    return calc