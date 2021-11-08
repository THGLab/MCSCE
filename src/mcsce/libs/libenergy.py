"""
Code for preparing and calculating various energy terms.

Original code in this file from IDP Conformer Generator package
(https://github.com/julie-forman-kay-lab/IDPConformerGenerator)
developed by Joao M. C. Teixeira (@joaomcteixeira), and added to the
MSCCE repository in commit 30e417937968f3c6ef09d8c06a22d54792297161.
Modifications herein are of MCSCE authors.
"""
# from mcsce import log
import numpy as np
import math
from numba import njit

from mcsce.core.definitions import vdW_radii_tsai_1999
from mcsce.core.build_definitions import (
    bonds_equal_3_inter,
    bonds_le_2_inter,
    bonds_equal_1_inter
    )
from mcsce.libs.libparse import extract_ff_params_for_seq
from mcsce.libs.libcalc import *


def prepare_energy_function(
        atom_labels,
        residue_numbers,
        residue_labels,
        forcefield,
        batch_size=4,
        terms=None,
        angle_term=True,
        dihedral_term=True,
        clash_term=True,
        lj_term=True,
        coulomb_term=True,
        gb_term=True,
        hpmf_term=True
        ):
    """Adapted from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira"""
    # Decide calculation terms from provided terms list
    if terms is not None:
        angle_term = "angle" in terms
        dihedral_term = "dihedral" in terms
        clash_term = "clash" in terms
        lj_term = "lj" in terms
        coulomb_term = "coulomb" in terms
        gb_term = "gb" in terms
        hpmf_term = "hpmf" in terms

    # this mask identifies covalently bonded pairs and pairs two bonds apart
    bonds_le_2_mask = create_bonds_apart_mask_for_ij_pairs(
        atom_labels,
        residue_numbers,
        residue_labels,
        forcefield.bonds_le2_intra,
        bonds_le_2_inter,
        base_bool=False,
        )

    # this mask identifies pairs exactly 3 bonds apart
    bonds_exact_3_mask = create_bonds_apart_mask_for_ij_pairs(
        atom_labels,
        residue_numbers,
        residue_labels,
        forcefield.bonds_eq3_intra,
        bonds_equal_3_inter,
        )

    bonds_1_mask = create_bonds_apart_mask_for_ij_pairs(
        atom_labels,
        residue_numbers,
        residue_labels,
        forcefield.res_topology,
        bonds_equal_1_inter,
        base_bool=False,
        )

    bonds_2_mask = (bonds_le_2_mask.astype(int) - bonds_1_mask.astype(int)).astype(bool)

    N = len(atom_labels)
    # Convert 2-bonds separated mask 1d array into the upper triangle of the 2d connecitivity matrix

    connectivity_matrix = np.zeros((N, N))
    connectivity_matrix[np.triu_indices(N, k=1)] = bonds_1_mask  # TODO: change to bonds_1_mask
    # The lower triangle can be mirrored from the upper triangle
    connectivity_matrix += connectivity_matrix.T
    # /
    # assemble energy function
    energy_func_terms_rij = [] # terms that are calculated from the pairwise distances of atoms
    energy_func_terms_coords = [] # terms that have to be calculated from the original coordinates

    if angle_term:
        angle_head_atom_matrix = np.zeros((N, N))
        angle_head_atom_matrix[np.triu_indices(N, k=1)] = bonds_2_mask
    else:
        angle_head_atom_matrix = None

    if dihedral_term:
        dihedral_head_atom_matrix = np.zeros((N, N))
        dihedral_head_atom_matrix[np.triu_indices(N, k=1)] = bonds_exact_3_mask
    else:
        dihedral_head_atom_matrix = None

    if angle_term or dihedral_term:
        from mcsce.libs.libbonded import prepare_angles_and_dihedrals, init_angle_calculator, init_dihedral_calculator
        angles, dihedrals, improper_dihedrals = prepare_angles_and_dihedrals(angle_head_atom_matrix,
                                                         dihedral_head_atom_matrix,
                                                         connectivity_matrix,
                                                         atom_labels,
                                                         residue_numbers,
                                                         residue_labels,
                                                         forcefield.forcefield)
    if angle_term:
        ang_calc = init_angle_calculator(angles)
        energy_func_terms_coords.append(ang_calc)
        # log.info('prepared angle')

    if dihedral_term:
        dih_calc = init_dihedral_calculator(dihedrals, improper_dihedrals)
        energy_func_terms_coords.append(dih_calc)
        # log.info('prepared dihedral')

    if clash_term:
        vdw_radii_sum = calc_vdw_radii_sum(atom_labels)
        vdw_radii_sum *= 0.6 # The clash check parameter as defined in the SI of the MCSCE paper
        vdw_radii_sum[bonds_1_mask] = 0
        vdw_radii_sum = vdw_radii_sum[None]
    else:
        vdw_radii_sum = None

    if lj_term:

        acoeff, bcoeff = create_LJ_params_raw(
            atom_labels,
            residue_numbers,
            residue_labels,
            forcefield.forcefield,
            )

        # 0.2 as 0.4
        _lj14scale = float(forcefield.forcefield['lj14scale'])
        acoeff[bonds_exact_3_mask] *= _lj14scale * 0.2
        bcoeff[bonds_exact_3_mask] *= _lj14scale * 0.2
        # acoeff[bonds_exact_3_mask] *= _lj14scale
        # bcoeff[bonds_exact_3_mask] *= _lj14scale
        acoeff[bonds_le_2_mask] = np.nan
        bcoeff[bonds_le_2_mask] = np.nan

        lf_calc = init_lennard_jones_calculator(acoeff[None], bcoeff[None])
        energy_func_terms_rij.append(lf_calc)
        # log.info('prepared lj')

    if coulomb_term or gb_term:

        charges_ij = create_Coulomb_params_raw(
            atom_labels,
            residue_numbers,
            residue_labels,
            forcefield.forcefield,
            )

    if coulomb_term:
        charges_ij_coulomb = charges_ij.copy()
        charges_ij_coulomb[bonds_exact_3_mask] *= float(forcefield.forcefield['coulomb14scale'])  # noqa: E501
        charges_ij_coulomb[bonds_le_2_mask] = np.nan

        coulomb_calc = init_coulomb_calculator(charges_ij_coulomb[None])
        energy_func_terms_rij.append(coulomb_calc)
        # log.info('prepared Coulomb')

    if gb_term:
        from mcsce.libs.libgb import create_atom_type_filters, init_gb_calculator
        atom_type_filters = create_atom_type_filters(atom_labels)
        gb_calc = init_gb_calculator(atom_type_filters, charges_ij[None])
        energy_func_terms_rij.append(gb_calc)
        # log.info('prepared GB implicit solvent')

    if hpmf_term:
        from mcsce.libs.libhpmf import convert_atom_types, init_hpmf_calculator
        atom_filter, sasa_atom_types = convert_atom_types(atom_labels, residue_numbers, residue_labels)
        # filter the connectivity matrix to only keep atoms that have SASA atom type definitions
        connectivity_matrix = connectivity_matrix[atom_filter, atom_filter]
        hpmf_calc = init_hpmf_calculator(sasa_atom_types, atom_filter, connectivity_matrix)
        energy_func_terms_coords.append(hpmf_calc)
        # log.info('prepared HPMF term')


    # in case there are iji terms, I need to add here another layer
    calc_energy = energycalculator_ij(
        calc_all_vs_all_dists,
        energy_func_terms_rij,
        energy_func_terms_coords,
        batch_size=batch_size,
        check_clash=clash_term,
        vdw_radii_sum=vdw_radii_sum
        )
    # log.info('done preparing energy func')
    return calc_energy

########################
## Create parameters
########################

def create_bonds_apart_mask_for_ij_pairs(
        atom_labels,
        residue_numbers,
        residue_labels,
        bonds_intra,
        bonds_inter,
        base_bool=False,
        ):
    """
    Create bool mask array identifying the pairs X bonds apart in ij pairs.

    Given `bonds_intra` and `bonds_inter` criteria, idenfities those ij
    atom pairs in N*(N-1)/2 condition (upper all vs all diagonal) that
    agree with the described bonds.

    Inter residue bonds are only considered for consecutive residues.

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

    Depends
    -------
    `gen_ij_pairs_upper_diagonal`
    `gen_atom_pair_connectivity_masks`

    Credit
    -------
    Borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira
    """
    atom_labels_ij_gen = gen_ij_pairs_upper_diagonal(atom_labels)
    residue_numbers_ij_gen = gen_ij_pairs_upper_diagonal(residue_numbers)
    residue_labels_ij_gen = gen_ij_pairs_upper_diagonal(residue_labels)

    bonds_indexes_gen = gen_atom_pair_connectivity_masks(
        residue_labels_ij_gen,
        residue_numbers_ij_gen,
        atom_labels_ij_gen,
        bonds_intra,
        bonds_inter,
        )

    num_ij_pairs = len(atom_labels) * (len(atom_labels) - 1) // 2
    other_bool = not base_bool
    bonds_mask = np.full(num_ij_pairs, base_bool)

    for idx in bonds_indexes_gen:
        bonds_mask[idx] = other_bool

    return bonds_mask

def create_LJ_params_raw(
        atom_labels,
        residue_numbers,
        residue_labels,
        force_field,
        ):
    """Create ACOEFF and BCOEFF parameters.
    Borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira"""
    sigmas_ii = extract_ff_params_for_seq(
        atom_labels,
        residue_numbers,
        residue_labels,
        force_field,
        'sigma',
        )

    epsilons_ii = extract_ff_params_for_seq(
        atom_labels,
        residue_numbers,
        residue_labels,
        force_field,
        'epsilon',
        )

    num_ij_pairs = len(atom_labels) * (len(atom_labels) - 1) // 2
    # sigmas
    sigmas_ij_pre = np.empty(num_ij_pairs, dtype=np.float64)
    sum_upper_diagonal_raw(np.array(sigmas_ii), sigmas_ij_pre)
    #
    # epsilons
    epsilons_ij_pre = np.empty(num_ij_pairs, dtype=np.float64)
    multiply_upper_diagonal_raw(
        np.array(epsilons_ii),
        epsilons_ij_pre,
        )
    #

    # mixing rules
    epsilons_ij = epsilons_ij_pre ** 0.5
    # mixing + nm to Angstrom converstion
    # / 2 and * 10
    sigmas_ij = sigmas_ij_pre * 5

    acoeff = 4 * epsilons_ij * (sigmas_ij ** 12)
    bcoeff = 4 * epsilons_ij * (sigmas_ij ** 6)
    # acoeff = epsilons_ij * (sigmas_ij ** 12)
    # bcoeff = 2 * epsilons_ij * (sigmas_ij ** 6)

    return acoeff, bcoeff

def create_Coulomb_params_raw(
        atom_labels,
        residue_numbers,
        residue_labels,
        force_field,
        ):
    """Borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira"""
    charges_i = extract_ff_params_for_seq(
        atom_labels,
        residue_numbers,
        residue_labels,
        force_field,
        'charge',
        )

    num_ij_pairs = len(atom_labels) * (len(atom_labels) - 1) // 2
    charges_ij = np.empty(num_ij_pairs, dtype=np.float64)
    multiply_upper_diagonal_raw(charges_i, charges_ij)
    charges_ij *= 0.25  # dielectic constant
    charges_ij *= 1389.35 # Coulomb's constant in unit of kJ.Angstrom/mol/e^2

    return charges_ij

def calc_vdw_radii_sum(atom_labels):
    """
    Calculate the van der Waals radii sum for atom pairs as for checking whether there are structure clashes
    """
    atom_types = [a[0] for a in atom_labels]
    vdw_radii = np.array([vdW_radii_tsai_1999[a] for a in atom_types])
    num_ij_pairs = len(atom_labels) * (len(atom_labels) - 1) // 2
    vdw_radii_sum_ij = np.empty(num_ij_pairs, dtype=np.float64)
    sum_upper_diagonal_raw(vdw_radii, vdw_radii_sum_ij)
    return vdw_radii_sum_ij

############################
## Other helper functions
############################

def are_connected(n1, n2, rn1, a1, a2, bonds_intra, bonds_inter):
    """
    Detect if a certain atom pair is bonded accordind to criteria.

    Considers only to the self residue and next residue

    Borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira
    """
    # requires
    assert isinstance(n1, int) and isinstance(n2, int), (type(n1), type(n2))
    assert all(isinstance(i, str) for i in (rn1, a1, a2)), \
        (type(i) for i in (rn1, a1, a2))
    assert all(isinstance(i, dict) for i in (bonds_intra, bonds_inter)), \
        (type(i) for i in (bonds_intra, bonds_inter))

    answer = (
        (n1 == n2 and a2 in bonds_intra[rn1][a1])
        or (
            n1 + 1 == n2
            and (
                a1 in bonds_inter  # this void KeyError
                and a2 in bonds_inter[a1]
                )
            )
        )

    assert isinstance(answer, bool)
    return answer



def gen_ij_pairs_upper_diagonal(data):
    """
    Generate upper diagonal ij pairs in tuples.

    The diagonal is not considered.

    Yields
    ------
    tuple of length 2
        IJ pairs in the form of N*(N-1) / 2.

    Credit
    -------
    Borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira
    """
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            yield (data[i], data[j])

def gen_atom_pair_connectivity_masks(
        res_names_ij,
        res_num_ij,
        atom_names_ij,
        connectivity_intra,
        connectivity_inter,
        ):
    """
    Generate atom pair connectivity indexes.

    Given atom information for the ij pairs and connectivity criteria,
    yields the index of the ij pair if the pair is connected according
    to the connectivity criteria.

    For example, if the ij pair is covalently bonded, or 3 bonds apart,
    etc.

    Parameters
    ----------
    res_names_ij
    res_num_ij,
    atom_names_ij, iterables of the same length and synchronized information.

    connectivity_intra,
    connectivity_inter, dictionaries mapping atom labels connectivity

    Depends
    -------
    `are_connected`

    Credit
    -------
    Borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira
    """
    zipit = zip(res_names_ij, res_num_ij, atom_names_ij)
    counter = 0
    for (rn1, _), (n1, n2), (a1, a2) in zipit:

        found_connectivity = are_connected(
            int(n1),
            int(n2),
            rn1,
            a1,
            a2,
            connectivity_intra,
            connectivity_inter,
            )

        if found_connectivity:
            yield counter

        counter += 1

########################
## Term calculators
########################

def init_lennard_jones_calculator(acoeff, bcoeff):
    """
    Calculate Lennard-Jones full pontential.

    The LJ potential is calculated fully and no approximations to
    proximity of infinite distance are considered.

    Parameters
    ----------
    acoeff, bcoeff : np.ndarray, shape (N, 3), dtype=np.float
        The LJ coefficients prepared already for the ij-pairs upon which
        the resulting function is expected to operate.
        IMPORTANT: it is up to the user to define the coefficients such
        that resulting energy is np.nan for non-relevant ij-pairs, for
        example, covalently bonded pairs, or pairs 2 bonds apart.



    Returns
    -------
    numba.njitted func
        Function closure with registered `acoeff`s and `bcoeff`s that
        expects an np.ndarray of distances with same shape as `acoeff`
        and `bcoeff`: (N,).
        `func` returns an integer.

    Credit
    -------
    Borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira
    """
    @njit
    def calc_lennard_jones(distances_ij, NANSUM=np.nansum):
        ar = acoeff / (distances_ij ** 12)
        br = bcoeff / (distances_ij ** 6)
        energy_ij = ar - br
        result = np.empty(distances_ij.shape[0])
        for i in range(distances_ij.shape[0]):
            result[i] = NANSUM(energy_ij[i])
        return result
    return calc_lennard_jones

def init_coulomb_calculator(charges_ij):
    """
    Calculate Coulomb portential.

    Parameters
    ----------
    charges_ij : np.ndarray, shape (N, 3), dtype=np.float
        The `charges_ij` prepared already for the ij-pairs upon which
        the resulting function is expected to operate.
        IMPORTANT: it is up to the user to define the charge such
        that resulting energy is np.nan for non-relevant ij-pairs, for
        example, covalently bonded pairs, or pairs 2 bonds apart.

    Returns
    -------
    numba.njitted func
        Function closure with registered `charges_ij` that expects an
        np.ndarray of distances with same shape as `acoeff` and `bcoeff`:
        (N,).
        `func` returns an integer.

    Credit
    -------
    Borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira
    """
    @njit
    def calculate(distances_ij, NANSUM=np.nansum):
        energy_ij = charges_ij / distances_ij
        result = np.empty(distances_ij.shape[0])
        for i in range(distances_ij.shape[0]):
            result[i] = NANSUM(energy_ij[i])
        return result
    return calculate

def energycalculator_ij(distf, efuncs_rij, efuncs_coords, batch_size=4, check_clash=False, vdw_radii_sum=None):
    """
    Calculate the sum of energy terms.

    This function works as a closure.

    Accepts energy terms that compute for non-redundant ij-pairs, and energy terms that directly take coordinates as inputs.

    Energy terms must have distances ij as unique positional parameter,
    and should return an integer.

    Example
    -------
    >>> ecalc = energycalculator_ij(calc_ij_pair_distances, [...], [...])
    >>> total_energy = ecalc(coords)

    Where the first `[...]` is a list containing energy term functions that take pairwise distances of atoms as input, and the second `[...]` is a list containing energy term functions that take raw coordinates as input.

    See Also
    --------
    init_lennard_jones_calculator
    init_coulomb_calculator

    Parameters
    ----------
    distf : func
        The function that will be used to calculate ij-pair distances
        on each call. If performance is a must, this function should be
        fast. `distf` function should receive `coords` as unique
        argument where `coords` is a np.ndarray of shape (N, 3), where N
        is the number of atoms, and 3 represents the XYZ coordinates.
        This function should return a np.ndarray of shape
        (N * (N - 1)) / 2,), dtype=np.float.

    efuncs_rij : list
        A list containing the energy terms functions. Energy term
        functions are prepared closures that accept the output of
        `distf` function.

    efuncs_coords : list
        A list containing the energy terms functions. Energy term
        functions are prepared closures that takes coordinates as input.

    check_clash : boolean
        When set to True, the calculator will first check whether the distance
        between any two atoms are smaller than the threshold (clash defined by
        0.8 * sum(VDW radii)). If clash exists, the energy will be infinity

    Returns
    -------
    func
        A function that accepts coords in the form of (N, 3). The
        coordinates sent to the resulting function MUST be aligned with
        the labels used to prepare the `efuncs` closures.

    Credit
    -------
    Adapted from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira, but added support for energy calculators that takes raw coordinates as input
    """
    def calculate(coords):
        if len(coords.shape) == 2:
            coords = coords[None] # make sure the first axis is the batch axis
        dist_ij = distf(coords)
        energies = np.empty(len(coords))
        if check_clash:
            clash_filter = (dist_ij < vdw_radii_sum).any(axis=1)
            energies[clash_filter] = np.inf
            dist_ij_noclash = dist_ij[~clash_filter]
            del dist_ij   # release memory to prevent OOM during parallel execution
            coords_noclash = coords[~clash_filter]
        else:
            clash_filter = np.zeros(len(coords)).astype(bool)
            dist_ij_noclash = dist_ij
            coords_noclash = coords
        energies_noclash = np.zeros(len(dist_ij_noclash))
        for i in range(math.ceil(len(energies_noclash) / batch_size)):
            batch_idx_start = i * batch_size
            batch_idx_end = (i + 1) * batch_size
            for func in efuncs_rij:
                energies_noclash[batch_idx_start: batch_idx_end] += func(dist_ij_noclash[batch_idx_start: batch_idx_end])
            for func in efuncs_coords:
                energies_noclash[batch_idx_start: batch_idx_end] += func(coords_noclash[batch_idx_start: batch_idx_end])
        energies[~clash_filter] = energies_noclash
        return energies
    return calculate



    # assert result.size == (data.size * data.size - data.size) // 2
    # assert abs(result[0] - data[0] * data[1]) < 0.0000001
    # assert abs(result[-1] - data[-2] * da
