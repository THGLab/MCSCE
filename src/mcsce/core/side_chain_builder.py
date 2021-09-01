"""
The main logic for building side chain with Monte Carlo Side Chain Ensemble algorithm, as defined in Bhowmick, Asmit, and Teresa Head-Gordon. "A Monte Carlo method for generating side chain structural ensembles." Structure 23.1 (2015): 44-55.

Coded by Jie Li
Date created: Jul 29, 2021
"""

from functools import partial
import numpy as np
from numba import njit

# from mcsce import log
from mcsce.core.rotamer_library import DunbrakRotamerLibrary
from mcsce.core.build_definitions import sidechain_templates
from mcsce.libs.libcalc import calc_torsion_angles, place_sidechain_template
from mcsce.libs.libscbuild import rotate_sidechain
from tqdm import tqdm
from copy import deepcopy

# from multiprocessing import set_start_method
# set_start_method("spawn")

_rotamer_library = DunbrakRotamerLibrary()

@njit
def choose_random(array):
    cumulative = [np.sum(array[:i + 1]) for i in range(len(array))]
    pointer = np.random.random() * cumulative[-1]
    for idx, num in enumerate(cumulative):
        if num > pointer:
            return idx

# def efunc_calculator_proxy(efunc, coord_input):
#     """
#     A helper function that takes an efunc object and the inputs, and returns the result from the efunc. 
#     Used as a proxy of the original efunc to make it picklable and used in multiprocessing
#     """
#     return efunc(coord_input)

class efunc_calculator_proxy:
    def __init__(self, efunc) -> None:
        self.efunc = efunc

    def __call__(self, coord_input):
        return self.efunc(coord_input)

def create_side_chain_structure(structure, sidechain_placeholders, energy_calculators, beta, save_addr=None):
    """
    A function that takes a backbone-only structure and generates a conformation using the Monte Carlo approach

    Parameters
    ----------
    structure: Structure object
        The structure with backbone atoms only

    sidechain_placeholders: list of Structure objects
        A list of structures with partially grown side chains

    energy_calculators: list of precompiled energy calculation functions
        A list of functions for calculating the energies of the generated conformations

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
    
    N_CA_C_coords = structure.get_sorted_minimal_backbone_coords()
    all_backbone_dihedrals = calc_torsion_angles(N_CA_C_coords)
    all_phi = np.concatenate([[np.nan], all_backbone_dihedrals[2::3]]) * 180 / np.pi
    all_psi = np.concatenate([all_backbone_dihedrals[::3], [np.nan]]) * 180 / np.pi
    for idx, resname in enumerate(structure.residue_types):
        # copy coordinates from the previous growing step to the current placeholder
        previous_coords = structure.coords
        structure = deepcopy(sidechain_placeholders[idx])
        template = sidechain_templates[resname]
        sidechain_atom_idx = template[1]
        n_sidechain_atoms = len(sidechain_atom_idx)
        new_coords = structure.coords
        new_coords[:-n_sidechain_atoms] = previous_coords
        structure.coords = new_coords
        coords = structure.coords
        if resname in ["GLY", "ALA"]:
            # For glycine and alanine, no need for deciding the conformation
            continue
        energy_func = energy_calculators[idx]
        # get all candidate conformations (rotamers) for this side chain
        candidiate_conformations, candidate_probs = _rotamer_library.retrieve_torsion_and_prob(resname, all_phi[idx], all_psi[idx])
        # perturb chi angles of the side chains by ~0.5 degrees
        candidiate_conformations += np.random.normal(scale=0.5, size=candidiate_conformations.shape)
        residue_bb_coords = N_CA_C_coords[idx * 3: (idx + 1) * 3]
        energies = []
        
        all_coords = np.tile(coords[None], (len(candidiate_conformations), 1, 1))
        # conformation_collection = []
        # for each rotamer, decide the resulting conformation and calculate its energy
        for tor_idx, tors in enumerate(candidiate_conformations):
            sidechain_with_specific_chi = rotate_sidechain(resname, tors)
            sc_conformation = place_sidechain_template(residue_bb_coords, sidechain_with_specific_chi[0])
            # conformation_collection.append(sc_conformation)
            all_coords[tor_idx, -n_sidechain_atoms:] = sc_conformation[sidechain_atom_idx]
            # energy = energy_func(coords)
            # energies.append(energy)
        energies = energy_func(all_coords)
        energies_raw = deepcopy(energies)  # This is the raw energy values
        # If all energies are inf, end this growth
        if np.isinf(energies).all():
            # log.info("Unresolvable clashes!")
            # np.save(f"{exec_idx}_{idx}.npy", all_coords)
            # np.save(f"{exec_idx}_tor.npy", candidiate_conformations)
            # structure.write_PDB(f"{exec_idx}.pdb")
            return structure, False, None, None
        # renormalize energies to avoid numerical issues
        energies -= min(energies)
        adjusted_weights = candidate_probs * np.exp(-beta * energies)
        selected_idx = choose_random(adjusted_weights)
        # set the coordinates of the actual structure that has side chain for this residue grown
        # coords[-n_sidechain_atoms:] = conformation_collection[selected_idx][sidechain_atom_idx]
        
        structure.coords = all_coords[selected_idx]
        energy = energies_raw[selected_idx] # The raw energy to be returned
    # all side chains have been created, then return the final structure
    if save_addr is not None:
        structure.write_PDB(save_addr)
    return structure, True, energy, save_addr

def create_side_chain(structure, n_trials, efunc_creator, temperature, parallel_worker=16):
    """
    Using the MCSCE workflow to add sidechains to a backbone-only PDB structure. The building process will be repeated for n_trial times, but only the lowest energy conformation will be returned 

    Parameters
    ----------
    structure: Structure object
        The structure with backbone atoms only

    n_trials: int
        The total number of trials for the generation procedure

    efunc_creator: partial function
        A partial function for creating energy functions for evaluating the generated conformations
        It should accept atom_label, res_num and res_label as inputs

    temperature: float
        The temperature value used for Boltzmann weighting

    parallel_worker: int
        Number of workers for parallel execution

    Returns
    ----------
    lowest_energy_conformation: Structure object
        The generated lowest-energy conformation structure with sidechains, or None when every trial of the conformation generation failed
    """
    # convert temperature to beta
    beta = 1 / (temperature * 0.008314462) # k unit: kJ/mol/K
    copied_backbone_structure = deepcopy(structure)
    sidechain_placeholder_list = []
    energy_calculator_list = []
    for idx, resname in tqdm(enumerate(structure.residue_types), total=len(structure.residue_types)):
        template = sidechain_templates[resname]
        structure.add_side_chain(idx + 1, deepcopy(template))
        sidechain_placeholder_list.append(deepcopy(structure))
        if resname not in ["GLY", "ALA"]:
            energy_func = efunc_creator(structure.atom_labels, 
                                        structure.res_nums,
                                        structure.res_labels)
            energy_calculator_list.append(energy_func)
        else:
            energy_calculator_list.append(None)
    conformations = []
    energies = []
    if parallel_worker == 1:
        for idx in tqdm(range(n_trials)):
            conf, succeeded, energy, _ = create_side_chain_structure(deepcopy(copied_backbone_structure), 
                            sidechain_placeholder_list, energy_calculator_list, beta, None)
            if succeeded:
                conformations.append(conf)
                energies.append(energy)
    else:
        import pathos
        pool = pathos.multiprocessing.ProcessPool(parallel_worker)

        result_iterator = pool.uimap(create_side_chain_structure, [deepcopy(copied_backbone_structure)] * n_trials, 
                            [sidechain_placeholder_list] * n_trials, [energy_calculator_list] * n_trials,
                            [beta] * n_trials, [None] * n_trials)
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


def create_side_chain_ensemble(structure, n_conformations, efunc_creator, temperature, save_path, parallel_worker=16):
    """
    Create a given number of conformation ensemble for the backbone-only structure of a protein

    Parameters
    ----------
    structure: Structure object
        The structure with backbone atoms only

    n_conformations: int
        The total number of conformations to be generated

    efunc_creator: partial function
        A partial function for creating energy functions for evaluating the generated conformations
        It should accept atom_label, res_num and res_label as inputs

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
    copied_backbone_structure = deepcopy(structure)
    sidechain_placeholder_list = []
    energy_calculator_list = []
    print("Start preparing energy calculators at different sidechain completion levels")
    for idx, resname in tqdm(enumerate(structure.residue_types), total=len(structure.residue_types)):
        template = sidechain_templates[resname]
        structure.add_side_chain(idx + 1, deepcopy(template))
        sidechain_placeholder_list.append(deepcopy(structure))
        if resname not in ["GLY", "ALA"]:
            energy_func = efunc_creator(structure.atom_labels, 
                                        structure.res_nums,
                                        structure.res_labels)
            # energy_func_proxy = partial(efunc_calculator_proxy, efunc=energy_func)
            # energy_func_proxy = efunc_calculator_proxy(energy_func)
            energy_calculator_list.append(energy_func)
        else:
            energy_calculator_list.append(None)
    print("Finished preparing all energy functions. Now start conformer generation")
    conformations = []
    all_success_count = 0
    if parallel_worker == 1:
        for idx in tqdm(range(n_conformations)):
            conf, succeeded = create_side_chain_structure(deepcopy(copied_backbone_structure), 
                            sidechain_placeholder_list, energy_calculator_list, beta, save_path + f"/{idx}.pdb")
            conformations.append(conf)
            all_success_count += int(succeeded)
    else:
        # import multiprocessing
        # pool = multiprocessing.Pool(processes=parallel_worker)

        import pathos
        pool = pathos.multiprocessing.ProcessPool(parallel_worker)
        # results = pool.starmap(create_side_chain_structure, [[deepcopy(copied_backbone_structure), 
        #                     sidechain_placeholder_list, energy_calculator_list, beta]] * n_conformations)
        result_iterator = pool.uimap(create_side_chain_structure, [deepcopy(copied_backbone_structure)] * n_conformations, 
                            [sidechain_placeholder_list] * n_conformations, [energy_calculator_list] * n_conformations,
                            [beta] * n_conformations, [save_path + f"/{n}.pdb" for n in range(n_conformations)])
        conformations = []
        success_indicator = []
        energies = {}
        all_success_count = 0
        for result in tqdm(result_iterator, total=n_conformations):
            conformations.append(result[0])
            success_indicator.append(result[1])
            if result[1]:
                # A succeeded case
                energies[result[3]] = result[2]
        with open(save_path + "/energies.csv", "w") as f:
            f.write("File name,Energy(kJ/mol)\n")
            for item in energies:
                f.write("%s,%f\n" % (item, energies[item]))
    return conformations, success_indicator

