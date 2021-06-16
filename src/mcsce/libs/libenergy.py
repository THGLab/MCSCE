from mcsce import log
import numpy as np
from numba import njit

from mcsce.core.build_definitions import (
    bonds_equal_3_inter,
    bonds_le_2_inter,
    bonds_equal_1_inter
    )
from mcsce.libs.libcalc import *
from mcsce.libs.libhpmf import *
from mcsce.libs.libgb import *

def prepare_energy_function(
        atom_labels,
        residue_numbers,
        residue_labels,
        forcefield,
        lj_term=True,
        coulomb_term=True,
        gb_term=True,
        hpmf_term=True
        ):
    """."""
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

    # /
    # assemble energy function
    energy_func_terms_rij = [] # terms that are calculated from the pairwise distances of atoms
    energy_func_terms_coords = [] # terms that have to be calculated from the original coordinates

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
        acoeff[bonds_le_2_mask] = np.nan
        bcoeff[bonds_le_2_mask] = np.nan

        lf_calc = init_lennard_jones_calculator(acoeff, bcoeff)
        energy_func_terms_rij.append(lf_calc)
        log.info('prepared lj')

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

        coulomb_calc = init_coulomb_calculator(charges_ij_coulomb)
        energy_func_terms_rij.append(coulomb_calc)
        log.info('prepared Coulomb')

    if gb_term:
        atom_type_filters = create_atom_type_filters(atom_labels)
        gb_calc = init_gb_calculator(atom_type_filters, charges_ij)
        energy_func_terms_rij.append(gb_calc)
        log.info('prepared GB implicit solvent')

    if hpmf_term:
        N = len(atom_labels)
        # Convert 2-bonds separated mask 1d array into the upper triangle of the 2d connecitivity matrix
        bonds_1_mask = create_bonds_apart_mask_for_ij_pairs(
            atom_labels,
            residue_numbers,
            residue_labels,
            forcefield.res_topology,
            bonds_equal_1_inter,
            base_bool=False,
        )
        connectivity_matrix = np.zeros((N, N))
        connectivity_matrix[np.triu_indices(N, k=1)] = bonds_1_mask  # TODO: change to bonds_1_mask
        # The lower triangle can be mirrored from the upper triangle
        connectivity_matrix += connectivity_matrix.T
        atom_filter, sasa_atom_types = convert_atom_types(atom_labels, residue_numbers, residue_labels)
        connectivity_matrix = connectivity_matrix[atom_filter, atom_filter]
        hpmf_calc = init_hpmf_calculator(sasa_atom_types, atom_filter, connectivity_matrix)
        energy_func_terms_coords.append(hpmf_calc)
        log.info('prepared HPMF term')


    # in case there are iji terms, I need to add here another layer
    calc_energy = energycalculator_ij(
        calc_all_vs_all_dists,
        energy_func_terms_rij,
        energy_func_terms_coords
        )
    log.info('done preparing energy func')
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
    """Create ACOEFF and BCOEFF parameters."""
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

    return acoeff, bcoeff

def create_Coulomb_params_raw(
        atom_labels,
        residue_numbers,
        residue_labels,
        force_field,
        ):
    """."""
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

    return charges_ij


############################
## Other helper functions
############################

def are_connected(n1, n2, rn1, a1, a2, bonds_intra, bonds_inter):
    """
    Detect if a certain atom pair is bonded accordind to criteria.

    Considers only to the self residue and next residue
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

def extract_ff_params_for_seq(
        atom_labels,
        residue_numbers,
        residue_labels,
        force_field,
        param,
        ):
    """
    Extract a parameter from forcefield dictionary for a given sequence.

    See Also
    --------
    create_conformer_labels

    Parameters
    ----------
    atom_labels, residue_numbers, residue_labels
        As returned by `:func:create_conformer_labels`.

    forcefield : dict

    param : str
        The param to extract from forcefield dictionary.
    """
    params_l = []
    params_append = params_l.append

    zipit = zip(atom_labels, residue_numbers, residue_labels)
    for atom_name, res_num, res_label in zipit:

        # adds C to the terminal residues
        if res_num == residue_numbers[-1]:
            res = 'C' + res_label
            was_in_C_terminal = True
            assert res.isupper() and len(res) == 4, res

        elif res_num == residue_numbers[0]:
            res = 'N' + res_label
            was_in_N_terminal = True
            assert res.isupper() and len(res) == 4, res

        else:
            res = res_label

        # TODO:
        # define protonation state in parameters
        if res_label.endswith('HIS'):
            res_label = res_label[:-3] + 'HIP'

        try:
            # force field atom type
            charge = force_field[res][atom_name]['charge']
            atype = force_field[res][atom_name]['type']

        # TODO:
        # try/catch is here to avoid problems with His...
        # for this purpose we are only using side-chains
        except KeyError:
            raise KeyError(tuple(force_field[res].keys()))

        if param in ["class", "element", "mass", "epsilon", "sigma"]:
            # These are parameters for non-specific atom types
            params_append(float(force_field[atype][param]))
        elif param == "charge":
            params_append(float(charge))

    assert was_in_C_terminal, \
        'The C terminal residue was never computed. It should have.'
    assert was_in_N_terminal, \
        'The N terminal residue was never computed. It should have.'

    assert isinstance(params_l, list)
    return params_l

def gen_ij_pairs_upper_diagonal(data):
    """
    Generate upper diagonal ij pairs in tuples.

    The diagonal is not considered.

    Yields
    ------
    tuple of length 2
        IJ pairs in the form of N*(N-1) / 2.
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
    """
    @njit
    def calc_lennard_jones(distances_ij, NANSUM=np.nansum):
        ar = acoeff / (distances_ij ** 12)
        br = bcoeff / (distances_ij ** 6)
        energy_ij = ar - br
        return NANSUM(energy_ij)
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
    """
    @njit
    def calculate(distances_ij, NANSUM=np.nansum):
        return NANSUM(charges_ij / distances_ij)
    return calculate

def energycalculator_ij(distf, efuncs_rij, efuncs_coords):
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

    Returns
    -------
    func
        A function that accepts coords in the form of (N, 3). The
        coordinates sent to the resulting function MUST be aligned with
        the labels used to prepare the `efuncs` closures.
    """
    def calculate(coords):
        dist_ij = distf(coords)
        energy = 0
        for func in efuncs_rij:
            energy += func(dist_ij)
        for func in efuncs_coords:
            energy += func(coords)
        return energy
    return calculate



    # assert result.size == (data.size * data.size - data.size) // 2
    # assert abs(result[0] - data[0] * data[1]) < 0.0000001
    # assert abs(result[-1] - data[-2] * da