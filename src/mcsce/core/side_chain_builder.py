"""
The main logic for building side chain with Monte Carlo Side Chain Ensemble algorithm,
 as defined in Bhowmick, Asmit, and Teresa Head-Gordon.
  "A Monte Carlo method for generating side chain structural ensembles."
  Structure 23.1 (2015): 44-55.

Coded by Jie Li
Date created: Jul 29, 2021
"""

import numpy as np
from numba import njit

from mcsce.core.rotamer_library import DunbrakRotamerLibrary
from mcsce.core.definitions import aa3to1
from mcsce.core.build_definitions import sidechain_templates
from mcsce.libs.libcalc import calc_torsion_angles, place_sidechain_template
from mcsce.libs.libscbuild import rotate_sidechain
from tqdm import tqdm
from copy import deepcopy
from functools import partial

from mcsce.libs.libstructure import Structure


_rotamer_library = DunbrakRotamerLibrary()

sidechain_placeholders = []
energy_calculators = []

@njit
def choose_random(array):
    cumulative = [np.sum(array[:i + 1]) for i in range(len(array))]
    pointer = np.random.random() * cumulative[-1]
    for idx, num in enumerate(cumulative):
        if num > pointer:
            return idx

def initialize_func_calc(efunc_creator, aa_seq=None, structure=None):
    """
    Helper function for initializing the energy function calculators according to the specified
    amino acid sequence or a structure object.

    Parameters
    ----------
    efunc_creator: partial function
        A partial function for creating energy functions for evaluating the generated conformations
        It should accept atom_label, res_num and res_label as inputs

    aa_seq: list
        List of 3-letter codes for the amino acid sequence

    structure: Structure object
        The structure with backbone atoms only
    """
    # declare global variabales
    global sidechain_placeholders
    global energy_calculators

    sidechain_placeholders = []
    energy_calculators = []
    print("Start preparing energy calculators at different sidechain completion levels")
    if structure is None:
        # create structure object according to the amino acid sequence
        fasta = "".join(aa3to1.get(res, 'X') for res in aa_seq)
        structure = Structure(fasta=fasta)
        structure.build()
    elif aa_seq is None:
        # extract amino acid sequence from structure object
        aa_seq = structure.residue_types
    structure = deepcopy(structure)
    sidechain_placeholders.append(deepcopy(structure))
    for idx, resname in tqdm(enumerate(aa_seq), total=len(aa_seq)):
        template = sidechain_templates[resname]
        structure.add_side_chain(idx + 1, template)
        sidechain_placeholders.append(deepcopy(structure))
        if resname not in ["GLY", "ALA"]:
            energy_func = efunc_creator(structure.atom_labels, 
                                        structure.res_nums,
                                        structure.res_labels)
            energy_calculators.append(energy_func)
        else:
            energy_calculators.append(None)
    print("Finished preparing all energy functions. Now start conformer generation")

def create_side_chain_structure(inputs):
    """
    A function that takes a backbone-only structure and generates a conformation using the Monte Carlo approach

    Parameters
    ----------
    (parameters are packaged into a list to allow multiprocessing)
    backbone_coords: np.array with shape (N, 3)
        Backbone atom coordinates that defines the conformation for growing side chains
        Order of atoms are assumed to be ["N", "CA", "C", "O", "H1", "H2", "H3"] for N-terminal residue,
        ["N", "CA", "C", "O", "H"] for middle residues, and 
        then ["N", "CA", "C", "O", "H", "OXT"] for C-terminal residue

    beta: float
        The beta value used for Boltzmann weighting

    save_addr: str or None
        The path for saving the generated structure. When None, the structure is not saved locally

    Returns
    ----------
    all_atom_structure: Structure object
        The structure including side chain atoms

    succeeded: bool
        Whether the side chain growth completed successfully

    energy:
        The energy for the generated conformation
    """
    backbone_coords, beta, save_addr = inputs
    assert len(sidechain_placeholders) > 0, "Energy functions have not yet initialized!"
    structure = deepcopy(sidechain_placeholders[0])
    structure.coords = backbone_coords
    N_CA_C_coords = structure.get_sorted_minimal_backbone_coords()
    all_backbone_dihedrals = calc_torsion_angles(N_CA_C_coords)
    all_phi = np.concatenate([[np.nan], all_backbone_dihedrals[2::3]]) * 180 / np.pi
    all_psi = np.concatenate([all_backbone_dihedrals[::3], [np.nan]]) * 180 / np.pi
    structure_coords = backbone_coords
    for idx, resname in enumerate(structure.residue_types):
        # copy coordinates from the previous growing step to the current placeholder
        previous_coords = structure_coords
        # structure = deepcopy(sidechain_placeholders[idx])
        template_struc, sidechain_atom_idx = sidechain_templates[resname]
        n_sidechain_atoms = len(sidechain_atom_idx)
        new_coords = sidechain_placeholders[idx + 1].coords
        new_coords[:-n_sidechain_atoms] = previous_coords
        structure_coords = new_coords
        # coords = structure.coords
        residue_bb_coords = N_CA_C_coords[idx * 3: (idx + 1) * 3]
        if resname in ["GLY", "ALA"]:
            # For glycine and alanine, no degrees of freedom for bond rotations, so just move side chain to appropriate position
            sc_conformation = place_sidechain_template(residue_bb_coords, template_struc.coords)
            structure_coords[-n_sidechain_atoms:] = sc_conformation[sidechain_atom_idx]
            continue
        energy_func = energy_calculators[idx]
        # get all candidate conformations (rotamers) for this side chain
        candidiate_conformations, candidate_probs = _rotamer_library.retrieve_torsion_and_prob(resname, all_phi[idx], all_psi[idx])
        # perturb chi angles of the side chains by ~0.5 degrees
        candidiate_conformations += np.random.normal(scale=0.5, size=candidiate_conformations.shape)
        energies = []
        
        all_coords = np.tile(structure_coords[None], (len(candidiate_conformations), 1, 1))
        # for each rotamer, decide the resulting conformation and calculate its energy
        for tor_idx, tors in enumerate(candidiate_conformations):
            sidechain_with_specific_chi = rotate_sidechain(resname, tors)  # THIS STEP MIGHT BE SLOW
            sc_conformation = place_sidechain_template(residue_bb_coords, sidechain_with_specific_chi[0])
            all_coords[tor_idx, -n_sidechain_atoms:] = sc_conformation[sidechain_atom_idx]
        energies = energy_func(all_coords)
        minimum_energy = min(energies)  # Keep track of the minimum energy so that the renormalized energies can be converted back
        
        # If all energies are inf, end this growth
        if np.isinf(energies).all():
            return None, False, None, None

        # renormalize energies to avoid numerical issues
        energies -= minimum_energy
        adjusted_weights = candidate_probs * np.exp(-beta * energies)
        selected_idx = choose_random(adjusted_weights)

        # set the coordinates of the actual structure that has side chain for this residue grown
        structure_coords = all_coords[selected_idx]
        energy = energies[selected_idx] + minimum_energy # The raw energy to be returned

    # all side chains have been created, then reorder all atoms and return the final structure
    structure = deepcopy(sidechain_placeholders[-1])
    structure.coords = structure_coords
    structure.reorder_with_resnum()
    if save_addr is not None:
        structure.write_PDB(save_addr)
    return structure, True, energy, save_addr

def create_side_chain(structure, n_trials, efunc_creator, temperature, parallel_worker=16, return_first_valid=False):
    """
    Using the MCSCE workflow to add sidechains to a backbone-only PDB structure. The building process will be repeated for n_trial times, but only the lowest energy conformation will be returned 

    Parameters
    ----------
    structure: Structure object
        The structure with backbone atoms only

    n_trials: int
        The total number of trials for the generation procedure
        if n_trials <=0, then sequentially generate structures until the first valid structure is generated

    efunc_creator: partial function
        A partial function for creating energy functions for evaluating the generated conformations
        It should accept atom_label, res_num and res_label as inputs

    temperature: float
        The temperature value used for Boltzmann weighting

    parallel_worker: int
        Number of workers for parallel execution

    return_first_valid: bool
        Controls the behavior of whether execute parallel building and return the first valid structure (no clashes), instead of generating a collection of structures and return the lowest energy one

    Returns
    ----------
    lowest_energy_conformation: Structure object
        The generated lowest-energy conformation structure with sidechains, or None when every trial of the conformation generation failed
    """
    # declare global variabales
    global copied_backbone_structure
    global sidechain_placeholders
    global energy_calculators
    
    # convert temperature to beta
    beta = 1 / (temperature * 0.008314462) # k unit: kJ/mol/K
    copied_backbone_structure = deepcopy(structure)
    sidechain_placeholders = []
    energy_calculators = []
    for idx, resname in tqdm(enumerate(structure.residue_types), total=len(structure.residue_types)):
        template = sidechain_templates[resname]
        structure.add_side_chain(idx + 1, template)
        sidechain_placeholders.append(deepcopy(structure))
        if resname not in ["GLY", "ALA"]:
            energy_func = efunc_creator(structure.atom_labels, 
                                        structure.res_nums,
                                        structure.res_labels)
            energy_calculators.append(energy_func)
        else:
            energy_calculators.append(None)
    conformations = []
    energies = []
    if return_first_valid:
        # Sequential execution with maximal n_trial times, but return the first valid structure
        for _ in range(n_trials):
            conf, succeeded, energy, _ = create_side_chain_structure([structure.coords, beta, None])
            if succeeded:
                return conf
        return None
    else:
        # Emsemble building with either sequential execution or parallelization
        if parallel_worker == 1:
            # sequential execution
            for idx in tqdm(range(n_trials)):
                conf, succeeded, energy, _ = create_side_chain_structure([structure.coords, beta, None])
                if succeeded:
                    conformations.append(conf)
                    energies.append(energy)
        else:
            import multiprocessing
            pool = multiprocessing.Pool(parallel_worker)

            result_iterator = pool.imap_unordered(
                create_side_chain_structure, \
                [[structure.coords, beta, None]] * n_trials)
            conformations = []
            energies = []
            for conf, succeeded, energy, _ in tqdm(result_iterator, total=n_trials):
                if succeeded:
                    conformations.append(conf)
                    energies.append(energy)
                
        # define the lowest energy conformation and return
        if len(energies) == 0:
            return None
        else:
            lowest_energy_idx = np.argmin(energies)
            return conformations[lowest_energy_idx]


def create_side_chain_ensemble(structure, n_conformations, temperature, save_path, parallel_worker=16):
    """
    Create a given number of conformation ensemble for the backbone-only structure of a protein

    Parameters
    ----------
    structure: Structure object
        The structure with backbone atoms only

    n_conformations: int
        The total number of conformations to be generated

    temperature: float
        The temperature value used for Boltzmann weighting

    save_path: str
        The folder path for saving all succeessfully generated PDB files

    Returns
    ----------
    conformations: list of Structure object
        The generated conformation structures

    all_success_count: int
        Total number of the generated structures that are not early-stopped due to unresolvable clashes
    """


    # convert temperature to beta
    beta = 1 / (temperature * 0.008314462) # k unit: kJ/mol/K
    
    conformations = []
    success_indicator = []
    energies = {}

    if parallel_worker == 1:
        for idx in tqdm(range(n_conformations)):
            conf, succeeded, energy, save_dir = create_side_chain_structure([structure.coords, beta, save_path + f"/{idx}.pdb"])
            conformations.append(conf)
            success_indicator.append(succeeded)
            if succeeded:
                energies[save_dir] = energy
    else:
        # import pathos
        import multiprocessing
        # pool = pathos.multiprocessing.ProcessPool(parallel_worker)
        pool = multiprocessing.Pool(parallel_worker)
        # result_iterator = pool.starmap(create_side_chain_structure, 
        # [[beta, save_path + f"/{n}.pdb"] for n in range(n_conformations)])
        # result_iterator = pool.uimap(create_side_chain_structure, 
        #                     [beta] * n_conformations, [save_path + f"/{n}.pdb" for n in range(n_conformations)])
        
        with tqdm(total=n_conformations) as pbar:
            for result in pool.imap_unordered(create_side_chain_structure,\
                [[structure.coords, beta, save_path + f"/{n}.pdb"] for n in range(n_conformations)]):
                conformations.append(result[0])
                success_indicator.append(result[1])
                if result[1]:
                    # A succeeded case
                    energies[result[3]] = result[2]
                pbar.update()

        pool.close()
        pool.join()
    with open(save_path + "/energies.csv", "w") as f:
        f.write("File name,Energy(kJ/mol)\n")
        for item in energies:
            f.write("%s,%f\n" % (item, energies[item]))
    return conformations, success_indicator

