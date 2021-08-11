from datetime import datetime
import os
from mcsce.core import build_definitions
from mcsce.libs.libbuild import *
from mcsce.libs.libstructure import Structure
from mcsce.libs.libenergy import prepare_energy_function
from mcsce.libs.libcalc import calc_torsion_angles
from mcsce.core.side_chain_builder import create_side_chain_ensemble
import argparse

from tqdm import tqdm


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("input_structure", help="A single PDB file for the backbone conformation, or a folder in which all PDB files will be processed")
parser.add_argument("n_conf", type=int, help="Number of side-chain conformations to generate (regardless of whether or not succeeded) for each given input backbone structure")
parser.add_argument("-w", "--n_worker", type=int, default=None, help="Number of parallel workers for executing side chain building. When not specified, use all CPUs installed in the machine")
parser.add_argument("-o", "--output_dir", default=None, help="The output position of generated PDB files. When not specified, it will be the name of the input file+'_mcsce'")
parser.add_argument("-l", "--log", default="log.csv", help="The log file save position")
args = parser.parse_args()


with open(args.log, "w") as f:
    f.write("PDB name,Succeeded,Time used\n ")

ff = build_definitions.forcefields["Amberff14SB"]
ff_obj = ff(add_OXT=True, add_Nterminal_H=True)

if args.n_worker is None:
    import multiprocessing
    n_worker = multiprocessing.cpu_count()
else:
    n_worker = args.n_worker

if args.input_structure[-3:].upper() == "PDB":
    all_pdbs = [args.input_structure]
else:
    input_folder = args.input_structure
    if not input_folder.endswith("/"):
        input_folder += "/"
    all_pdbs = [input_folder + f for f in os.listdir(input_folder) if f[-3:].upper() == "PDB"]

for f in all_pdbs:
    print("Now working on", f)
    t0 = datetime.now()
    if args.output_dir is None:
        output_dir = os.path.splitext(os.path.basename(f))[0] + "_mcsce"
    else:
        output_dir = args.output_dir + "/" + os.path.splitext(os.path.basename(f))[0] + "_mcsce"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    s = Structure(f)
    s.build()
    s = s.remove_side_chains()


    conformations, success_indicator = create_side_chain_ensemble(s, args.n_conf, partial(prepare_energy_function,
             forcefield=ff_obj, terms=["lj", "clash"]), temperature=300, save_path=output_dir, parallel_worker=n_worker)

    with open(args.log, "a") as logfile:
        logfile.write("%s,%d,%s\n" % (f, sum(success_indicator), str(datetime.now() - t0)))

print("All finished!")
