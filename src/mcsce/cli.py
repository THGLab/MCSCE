import argparse
import os
from datetime import datetime


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("input_structure", help="A single PDB file for the backbone conformation, or a folder in which all PDB files will be processed")
parser.add_argument("n_conf", type=int, help="Number of side-chain conformations to generate (regardless of whether or not succeeded) for each given input backbone structure")
parser.add_argument("-w", "--n_worker", type=int, default=None, help="Number of parallel workers for executing side chain building. When not specified, use all CPUs installed in the machine")
parser.add_argument("-o", "--output_dir", default=None, help="The output position of generated PDB files. When not specified, it will be the name of the input file+'_mcsce'")
parser.add_argument("-s", "--same_structure", action="store_true", default=False, help="When generating PDB files in a folder, whether each structure in the folder has the same amino acid sequence. When this is set to True, the energy function preparation will be done only once.")
parser.add_argument("-b", "--batch_size", default=16, type=int, help="The batch size used for calculating energies for conformations. Consider decreasing batch size when encountered OOM in building.")
parser.add_argument("-m", "--mode", choices=["ensemble", "simple", "exhaustive"], default="ensemble", help="This option controls whether a structural ensemble or just a single structure is created for every input structure. The default behavior is to create a structural ensemble. Simple/exhaustive modes are for creating single structure. In simple mode, n_conf structures are tried sequentially and the first valid structure is returned, and in exhaustive mode, a total number of n_conf structures will be created and the lowest energy conformation is returned.")
parser.add_argument("-l", "--logfile", default="log.csv", help="File name of the log file")


def load_args(parser):
    """Load argparse commands."""
    return parser.parse_args()


def cli(parser, main):
    """Command-line interface entry-point."""
    cmd = load_args(parser)
    main(**vars(cmd))


def maincli():
    """Independent client entry point."""
    cli(parser, main)

def read_structure_and_check(file_name):
    """
    Helper function for reading a structure from a given filename and returns the structure object
    Also checks whether there are missing atoms in the structure
    """
    from mcsce.libs.libstructure import Structure
    s = Structure(file_name)
    s.build()
    missing_backbone_atoms = s.check_backbone_atom_completeness()
    if len(missing_backbone_atoms) > 0:
        message = f"WARNING! These atoms are missing from the current backbone structure [{file_name}]:"
        for resid, atom_name in missing_backbone_atoms:
            message += f"\n{resid} {atom_name}"
        print(message + "\n")
    return s


def main(input_structure, n_conf, n_worker, output_dir, logfile, mode, batch_size=4, same_structure=False):

    # antipattern to save time
    from tqdm.std import tqdm
    from mcsce.core.side_chain_builder import initialize_func_calc, create_side_chain_ensemble, create_side_chain
    from mcsce.core.build_definitions import forcefields
    from mcsce.libs.libenergy import prepare_energy_function
    from mcsce.libs.libstructure import Structure
    from functools import partial
    base_dir = output_dir

    with open(logfile, "w") as f:
        f.write("PDB name,Succeeded,Time used\n")

    ff = forcefields["Amberff14SB"]
    ff_obj = ff(add_OXT=True, add_Nterminal_H=True)

    if n_worker is None:
        import multiprocessing
        n_worker = multiprocessing.cpu_count() - 1
    else:
        n_worker = n_worker
    print("# workers:", n_worker)
    print("mode: ", mode)
    
    if input_structure[-3:].upper() == "PDB":
        all_pdbs = [input_structure]
    else:
        input_folder = input_structure
        if not input_folder.endswith("/"):
            input_folder += "/"
        all_pdbs = [input_folder + f for f in os.listdir(input_folder) if f[-3:].upper() == "PDB"]

    if same_structure:
        # Assume all structures in a folder are the same: the energy creation step can be done only once
        s = Structure(all_pdbs[0])
        s.build()
        s = s.remove_side_chains()
        initialize_func_calc(partial(prepare_energy_function, batch_size=batch_size,
         forcefield=ff_obj, terms=["lj", "clash"]),
        structure=s)
    if mode == "simple" and same_structure and n_worker > 1:
        # parallel executing sequential trials on the same structure (different conformations)
        t0 = datetime.now()
        structures = [read_structure_and_check(f) for f in all_pdbs]
        side_chain_parallel_creator = partial(create_side_chain, 
                                              n_trials=n_conf,
                                              temperature=300,
                                              parallel_worker=1,
                                              return_first_valid=True)
        import multiprocessing
        pool = multiprocessing.Pool(n_worker)

        with tqdm(total=len(structures)) as pbar:
            for original_filename, best_structure in \
                zip(all_pdbs, pool.imap(side_chain_parallel_creator, structures)):
                if best_structure is not None:
                    if base_dir is None:
                        save_name = os.path.splitext(os.path.basename(original_filename))[0] + "_mcsce.pdb"
                    else:
                        save_name = base_dir + "/" + os.path.splitext(os.path.basename(original_filename))[0] + "_mcsce.pdb"
                    best_structure.write_PDB(save_name)
                    with open(logfile, "a") as _log:
                        _log.write("%s,%d,%s\n" % (original_filename, 1, str(datetime.now() - t0)))
                else:
                    with open(logfile, "a") as _log:
                        _log.write("%s,%d,%s\n" % (original_filename, 0, str(datetime.now() - t0)))
                pbar.update()
        pool.close()
        pool.join()

    else:
        for f in all_pdbs:
            print("Now working on", f)
            t0 = datetime.now()
            if base_dir is None:
                output_dir = os.path.splitext(os.path.basename(f))[0] + "_mcsce"
            else:
                output_dir = base_dir + "/" + os.path.splitext(os.path.basename(f))[0] + "_mcsce"
            if not os.path.exists(output_dir) and mode == "ensemble":
                os.makedirs(output_dir)
            s = read_structure_and_check(f)
            

            if not same_structure:
                initialize_func_calc(partial(prepare_energy_function, batch_size=batch_size,
                forcefield=ff_obj, terms=["lj", "clash"]), structure=s)

            if mode == "ensemble":
                conformations, success_indicator = create_side_chain_ensemble(
                    s,
                    n_conf,
                    temperature=300,
                    save_path=output_dir,
                    parallel_worker=n_worker,
                    )

                with open(logfile, "a") as _log:
                    _log.write("%s,%d,%s\n" % (f, sum(success_indicator), str(datetime.now() - t0)))
            else:
                best_structure = create_side_chain(s, n_conf, 300, n_worker, mode=="simple")
                if best_structure is not None:
                    best_structure.write_PDB(output_dir + ".pdb")
                    with open(logfile, "a") as _log:
                        _log.write("%s,%d,%s\n" % (f, 1, str(datetime.now() - t0)))
                else:
                    with open(logfile, "a") as _log:
                        _log.write("%s,%d,%s\n" % (f, 0, str(datetime.now() - t0)))

    print("All finished!")


if __name__ == '__main__':
    maincli()
