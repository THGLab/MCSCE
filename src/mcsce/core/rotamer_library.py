"""
This file defines the function that reads the Dunbrack protein side chain rotamer library

Shapovalov, Maxim V., and Roland L. Dunbrack Jr. "A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions." Structure 19.6 (2011): 844-858.

Coded by Jie Li
Date created: Jul 28, 2021
"""

from pathlib import Path
import numpy as np

_filepath = Path(__file__).resolve().parent  # folder
_library_path = _filepath.joinpath('data', 'SimpleOpt1-5', 'ALL.bbdep.rotamers.lib')

def get_closest_angle(input_angle):
    """
    Find closest angle in [-180, +170] with 10 degree separations
    """
    if input_angle > 175:
        # the closest would be 180, namely -180
        return -180
    return round(input_angle / 10) * 10

class DunbrakRotamerLibrary:
    """
    data structure: {(restype, psi, phi): [np.array<N, c>(N: number of rotamers, c: number of chi values, the rotamer values in degree), np.array<N>(probability)]}
    """
    def __init__(self, probability_threshold=0.001, augment_with_std=True) -> None:
        """
        probability_threshold: the minimum probability of a rotamer to be considered
        augment_with_std: when set to True, the chi_(1,2) +- sigma are also taken as individual rotamers, and 
        the probabilities for chi_i, chi_i + sigma, chi_i - sigma becomes 1/9 of the original probability of chi_i,
        following the implementation in Bhowmick, Asmit, and Teresa Head-Gordon. "A Monte Carlo method for generating side chain structural ensembles." Structure 23.1 (2015): 44-55.
        """
        self._data = {}
        with open(_library_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    # comment line to be ignored
                    continue
                restype, phi, psi, _, _, _, _, _, prob, chi1, chi2, chi3, chi4, chi1_sigma, chi2_sigma, chi3_sigma, chi4_sigma = \
                    line.split()
                prob = float(prob)
                if prob > probability_threshold:
                    phi = int(phi)
                    psi = int(psi)
                    if phi == 180 or psi == 180:
                        # the -180 angle data should be the same
                        assert (restype, min(phi, -180), min(psi, -180)) in self._data
                        continue
                    chi1 = float(chi1)
                    chi2 = float(chi2)
                    chi3 = float(chi3)
                    chi4 = float(chi4)
                    chi1_sigma = float(chi1_sigma)
                    chi2_sigma = float(chi2_sigma)
                    chi3_sigma = float(chi3_sigma)
                    chi4_sigma = float(chi4_sigma)
                    chi_arr = np.array([[chi1, chi2, chi3, chi4]])
                    current_probability = [prob]
                    if augment_with_std:
                        if restype in ["CYS", "SER", "VAL"]:
                            # these residues only have chi1
                            chi_arr = np.array([[new_chi1, chi2, chi3, chi4] 
                            for new_chi1 in [chi1 - chi1_sigma, chi1, chi1 + chi1_sigma]])
                            current_probability = [prob / 3] * 3
                        else:
                            chi_arr = np.array([[new_chi1, new_chi2, chi3, chi4]
                                    for new_chi1 in [chi1 - chi1_sigma, chi1, chi1 + chi1_sigma]
                                    for new_chi2 in [chi2 - chi2_sigma, chi2, chi2 + chi2_sigma]])
                            current_probability = [prob / 9] * 9
                    label = (restype, phi, psi)
                    if label in self._data:
                        self._data[label][0].append(chi_arr)
                        self._data[label][1].append(current_probability)
                    else:
                        self._data[label] = [[chi_arr], [current_probability]]
        for item in self._data:
            chis = np.concatenate(self._data[item][0])
            if item[0] in ["CYS", "SER", "THR", "VAL"]: # residues with only chi1
                chis = chis[:, :1]
            elif item[0] in ["ASN", "ASP", "HIS", "ILE", "LEU", "PHE", "PRO", "TRP", "TYR"]:
                # residues with chi1 and chi2
                chis = chis[:, :2]
            elif item[0] in ["GLN", "GLU", "MET"]:
                # residues with chi1 to chi3
                chis = chis[:, :3]
            probabilities = np.concatenate(self._data[item][1])
            self._data[item] = [chis, probabilities]

    def retrieve_torsion_and_prob(self, residue_type, phi, psi):
        if residue_type in ["HID", "HIE", "HIP"]:
            residue_type = "HIS"
        # TODO: select phi/psi according to the Ramanchandran plot
        if np.isnan(phi):
            # This is the first residue, which do not have phi, so select a random phi
            phi = np.random.uniform(-180, 180)
        if np.isnan(psi):
            # this is the last residue, which do not have psi, so select a random psi
            psi = np.random.uniform(-180, 180)
        # otherwise, return the library contents corresponding to closest phi and psi
        return self._data[(residue_type, get_closest_angle(phi), get_closest_angle(psi))]

if __name__ == "__main__":
    library = DunbrakRotamerLibrary()
    print(library.retrieve_torsion_and_prob("ARG", 73, -54.8))