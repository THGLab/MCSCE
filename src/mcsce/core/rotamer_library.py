"""
This file defines the function that reads the Dunbrack protein side chain rotamer library

Shapovalov, Maxim V., and Roland L. Dunbrack Jr. "A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions." Structure 19.6 (2011): 844-858.

Coded by Jie Li
Date created: Jul 28, 2021
Modified by Oufan Zhang to add ptm rotamers
"""

from pathlib import Path
import numpy as np
from mcsce.core.definitions import ptm_aa, ptm_h

_filepath = Path(__file__).resolve().parent  # folder
_library_path = _filepath.joinpath('data', 'SimpleOpt1-5', 'ALL.bbdep.rotamers.lib')
_ptm_library_path = _filepath.joinpath('data', 'ptm_rotamer.lib')


def wrap_angles(rot_angle):
    more_mask = rot_angle > 180.
    rot_angle[more_mask] -= 360.
    less_mask = rot_angle < -180.
    rot_angle[less_mask] += 360.
    return rot_angle

def get_closest_angle(input_angle, degsep=10):
    """
    Find closest angle in [-180, +170] with n degree separations
    """
    if input_angle > 180 - degsep/2.:
        # the closest would be 180, namely -180
        return -180
    return round(input_angle/degsep)*degsep 

def get_floor_angle(input_angle, degsep=120):
    """
    Round to angle in [0, 360) at the floor of n degree separations
    """
    if input_angle == 360.:
        return 0
    return np.floor(input_angle/degsep)*degsep


def sample_torsion(data, nchis):
    assert data.shape[1] == nchis*2
    vals = data[:, :nchis]
    sigs = data[:, -nchis:]
    return np.random.normal(vals, sigs)


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
        self.prob_threshold = probability_threshold
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
                    append_current_prob = True
                    if augment_with_std:
                        if restype in ["CYS", "SER", "VAL"]:
                            # these residues only have chi1
                            new_prob = prob / 3
                            if new_prob > probability_threshold:
                                chi_arr = np.array([[new_chi1, chi2, chi3, chi4] 
                                for new_chi1 in [chi1 - chi1_sigma, chi1, chi1 + chi1_sigma]])
                                current_probability = [new_prob] * 3
                            else:
                                append_current_prob = False
                        else:
                            new_prob = prob / 9
                            if new_prob > probability_threshold:
                                chi_arr = np.array([[new_chi1, new_chi2, chi3, chi4]
                                        for new_chi1 in [chi1 - chi1_sigma, chi1, chi1 + chi1_sigma]
                                        for new_chi2 in [chi2 - chi2_sigma, chi2, chi2 + chi2_sigma]])
                                current_probability = [new_prob] * 9
                            else:
                                append_current_prob = False
                    if append_current_prob:
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
            self._data[item] = [wrap_angles(chis), probabilities]

    def retrieve_torsion_and_prob(self, residue_type, phi, psi, ptmlib):
        if np.isnan(phi):
            # This is the first residue, which do not have phi, so use phi=-180 as default
            phi = -180
        if np.isnan(psi):
            # this is the last residue, which do not have psi, so use psi=-180 as default
            psi = -180

        if residue_type in ptm_aa:
            chis, probs = self._data[(ptm_aa[residue_type], get_closest_angle(phi), get_closest_angle(psi))]
            # phosphate protonation states
            if residue_type in ['S1P', 'T1P', 'Y1P', 'H1D', 'H1E', 'H2E']:
                residue_type = ptm_h[residue_type]
            if residue_type not in ptmlib._info:
                print("ptm residue rotamers not provided, assumes rotamers of unmodified residue")
                return [chis, probs]

            # search ptm library
            ptm_info = ptmlib.get_dependence(residue_type)
            nchis = ptm_info[1]
 
            if ptm_info[-1] == 0:
                ptm_data = ptmlib.retrieve_torsion_and_prob(residue_type, -360.)
                ptm_probs = np.array(ptm_data)[:, 0]
                torsions = wrap_angles(sample_torsion(np.array(ptm_data)[:, 1:], nchis))
                assert torsions.shape[1] == ptm_info[0]
                return [torsions, ptm_probs]
          
            dchis = chis[:, ptm_info[-1]-1]
            new_chis = []
            new_probs = []
            for i in range(chis.shape[0]):
                ptm_data = ptmlib.retrieve_torsion_and_prob(residue_type, dchis[i])
                ptm_probs = np.array(ptm_data)[:, 0]
                torsions = wrap_angles(sample_torsion(np.array(ptm_data)[:, 1:], nchis))
                for j in range(len(ptm_probs)):
                    p = probs[i]*ptm_probs[j] 
                    if p > self.prob_threshold:
                        new_chis.append(np.concatenate((chis[i], torsions[j])))
                        new_probs.append(p)
            assert len(new_chis[0]) == ptm_info[0]
            return [np.array(new_chis), np.array(new_probs)]

        if residue_type in ["HID", "HIE", "HIP"]:
            residue_type = "HIS"

        return self._data[(residue_type, get_closest_angle(phi), get_closest_angle(psi))]


class ptmRotamerLib():
    """
    data structure: {(restype, depended chi): [np.array<N, c>(N: number of rotamers, c: (probability, rotamer values, sigma values in degree)]}
    info structure: {(restype: [total chis, number of additional chis, dependence])}
    """
    def __init__(self, probability_threshold=0.001):
        self._data = {}
        self._info = {}
        with open(_ptm_library_path) as f:
            for line in f:
                line = line.strip()
                # read chi information
                if len(line) == 5 and line.startswith("# "):
                    restype = line.split()[-1]
                    self._info[restype] = []
                    continue
                if line.startswith("# Number of chi"):
                    self._info[restype].append(int(line.split()[-1]))
                    continue
                if line.startswith("# Number of additional chi"):
                    self._info[restype].append(int(line.split()[-1]))
                    continue
                if line.startswith("# Dependence"):
                    self._info[restype].append(int(line.split()[-1]))
                    continue
                if line.startswith("# No Dependence"):
                    self._info[restype].append(0)
                    continue
                if line.startswith("#") or len(line) == 0:
                    # other commented line
                    continue
                # read in rotamers
                if line.startswith(restype):
                    items = line.split()
                    if self._info[restype][-1] > 0:
                        # with dependence
                        chid = float(items[1]) 
                        label = (restype, chid)
                        if label in self._data:
                            self._data[label].append([float(n) for n in items[3:]])
                        else:
                            self._data[label] = [[float(n) for n in items[3:]]]
                    else:
                        chid = -360.
                        label = (restype, chid) 
                        if label in self._data:
                            self._data[label].append([float(n) for n in items[1:]])
                        else:
                            self._data[label] = [[float(n) for n in items[1:]]]
                    #if restype == 'SEP': print(label, self._data[label])    
 
    def get_dependence(self, residue_type):
        return self._info[residue_type]

    def retrieve_torsion_and_prob(self, residue_type, dchi=-360.):
        if self._info[residue_type][-1] == 0:
            return self._data[(residue_type, -360.)]
        else:
            # convert to [0, 360) scale
            if dchi < 0: dchi += 360.
            rchi = get_floor_angle(dchi, 120.)
            return self._data[(residue_type, rchi)]


if __name__ == "__main__":
    library = DunbrakRotamerLibrary()
    ptmlib = ptmRotamerLib()
    print(library.retrieve_torsion_and_prob("S1P", 73, -54.8, ptmlib))
