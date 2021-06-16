"""
Tools for calculating generalized Born term in the energy function:

Lin, Matthew S., Nicolas Lux Fawzi, and Teresa Head-Gordon. "Hydrophobic potential of mean force as a solvation function for protein structure prediction." Structure 15.6 (2007): 727-740.

The Born radii will be evaluated with a neural network, according to:

Mahmoud, Saida Saad Mohamed, et al. "Generalized Born radii computation using linear models and neural networks." Bioinformatics 36.6 (2020): 1757-1764.

Coded by Jie Li
Jun 14, 2021
"""

from keras.models import model_from_json
import numpy as np
import numba

from mcsce.libs.libcalc import multiply_upper_diagonal_raw



@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin <= 0 or bin >= n:
        return None
    else:
        return bin


@numba.jit(nopython=True)
def calc_histogram(a):
    """
    A numba-accelerated helper function that calculates histogram within range of (0, 16) with bin size of 0.2. Expect a 2d array and only does binning on the last axis
    """
    hist = np.zeros((a.shape[0], 80), dtype=np.intp)
    bin_edges = np.arange(0, 16.1, 0.2)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            x = a[i][j]
            bin = compute_bin(x, bin_edges)
            if bin is not None:
                hist[i, int(bin)] += 1
    return hist


class BornRadiiNNPredictor:
    def __init__(self) -> None:
        self.models = {}
        for atom in ["C", "H", "O", "N", "S"]:
            with open(f"iGBR_NN/model_{atom}.json") as f:
                model_json = f.read()
            model = model_from_json(model_json)
            model.load_weights(f"iGBR_NN/model_{atom}.h5")
            self.models[atom] = model

    def calc_born_radii(self, dist_upper_triangle, filters):
        """
        Main calculator that takes a pairwise distance array and the atom type filters, and uses NN that takes input bin counts for various atom types to predict the inverse born radii for each atom 

        Parameters
        ---------
        dist_upper_triangle : flattened upper triangle of a pairwise distance matrix
        filters : a dictionary of boolean atom type filters, output of the function `create_atom_type_filters`

        Returns
        ---------
        a 1d float array for the calculated Born radii for each atom
        """
        N = filters["C"].shape[0]
        dist_mat = np.zeros((N, N))
        dist_mat[np.triu_indices(N, k=1)] = dist_upper_triangle  # TODO: change to bonds_1_mask
        # The lower triangle can be mirrored from the upper triangle
        dist_mat += dist_mat.T
        output_arr = np.zeros(N)
        for atom in ["C", "H", "O", "N", "S"]:
            atom_dist = dist_mat[filters[atom]]
            input_arr = np.concatenate([calc_histogram(atom_dist[:, filters[atom_]]) for atom_ in ["H", "O", "N", "C", "S"]], axis=1)
            inv_born_radii = self.models[atom].predict(input_arr).flatten()
            output_arr[filters[atom]] = inv_born_radii
        output_arr = 1 / output_arr
        return output_arr



def create_atom_type_filters(atom_labels):
    """
    Function for creating atom type filters according to C, H, O, N, S so that the atomic distance bins for different atom types can be calculated more efficienctly.

    Paramters
    ---------
    atom_labels : iterable, list or np.ndarray
        The protein atom labels. Ex: ['N', 'CA, 'C', 'O', 'CB', ...]

    Returns
    ---------
    dictionary of {atom_type: filter}, atom_type is one of {C, H, O, N, S}, and filter is a boolean array representing the atom type filter for that atom type
    """
    filters = {}
    for atom in ["C", "H", "O", "N", "S"]:
        filters[atom] = np.char.startswith(atom_labels, atom)
    return filters


def init_gb_calculator(atom_filters, charges_ij, ep_p=4.0, ep_w=80.0):
    """
    Calculate generalized Born implicit solvent term

    Parameters
    ----------
        atom_filters : a dictionary of boolean atom type filters {atom_type: filter}, atom_type is one of {C, H, O, N, S}, and filter is a boolean array representing the atom type filter for that atom type

        charges_ij : np.ndarray, shape N*(N-1)/2, dtype=np.float
        The `charges_ij` prepared already for the ij-pairs upon which
        the resulting function is expected to operate.
        IMPORTANT: it is up to the user to define the charge such
        that resulting energy is np.nan for non-relevant ij-pairs, for
        example, covalently bonded pairs, or pairs 2 bonds apart.

        ep_p : dielectric constant for the protein

        ep_w : dielectric constant for water

    Returns
    ----------
    Function closure with registered `charges_ij` that expects an
        np.ndarray of distances with shape N*(N-1)/2
    """
    Ri_predictor = BornRadiiNNPredictor()
    coef = -0.5 * (1 / ep_p - 1 / ep_w)

    def calculate(r_ij):
        # First calculate generalized Born radii
        R_i = Ri_predictor.calc_born_radii(r_ij, atom_filters)
        R_ij = np.empty_like(r_ij, dtype=np.float64)
        multiply_upper_diagonal_raw(R_i, R_ij) # create pairwise Born radii products
        f_ij = np.sqrt(r_ij ** 2 + R_ij * np.exp(-r_ij ** 2 / (4 * R_ij)))
        V_GB = coef * np.sum(charges_ij / f_ij)
        return V_GB

    return calculate

