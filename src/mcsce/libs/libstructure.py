"""
Store internal protein structure representation.

Classes
-------
Structure
    The main API that represents a protein structure in MCSCE.

Original code in this file from IDP Conformer Generator package
(https://github.com/julie-forman-kay-lab/IDPConformerGenerator)
developed by Joao M. C. Teixeira (@joaomcteixeira), and added to the
MSCCE repository in commit 30e417937968f3c6ef09d8c06a22d54792297161.
Modifications herein are of MCSCE authors.
"""
import traceback
import warnings
from collections import defaultdict
from functools import reduce

import numpy as np
from copy import deepcopy
# from mcsce import log
from mcsce.core.definitions import backbone_atoms, aa3to1
from mcsce.core.definitions import residue_elements as _allowed_elements
from mcsce.core.exceptions import (
    EmptyFilterError,
    NotBuiltError,
    ParserNotFoundError,
    PDBFormatError,
    )
from mcsce.libs import libpdb
from mcsce.libs.libcif import CIFParser, is_cif
from mcsce.libs.libparse import group_runs, sample_case, type2string, is_valid_fasta, translate_seq_to_3l
from mcsce.libs.libpdb import RE_ENDMDL, RE_MODEL, delete_insertions
from mcsce.libs.libcalc import place_sidechain_template
# from idpconfgen.logger import S


class Structure:
    """
    Hold structural data from PDB/mmCIF files.

    Run the ``.build()`` method to read the structure.

    Cases for PDB Files:
    * If there are several MODELS only the first model is considered.

    Parameters
    ----------
    data : str, bytes, Path
        Raw structural data from PDB/mmCIF formatted files.

    Examples
    --------
    Opens a PDB file, selects only chain 'A' and saves selection to a file.
    >>> s = Structure('1ABC.pdb')
    >>> s.build()
    >>> s.add_filter_chain('A')
    >>> s.write_PDB('out.pdb')

    Opens a mmCIF file, selects only residues above 50 and saves
    selection to a file.
    >>> s = Structure('1ABC.cif')
    >>> s.build()
    >>> s.add_filter(lambda x: int(x[col_resSeq]) > 50)
    >>> s.write_PDB('out.pdb')
    """

    __slots__ = [
        '_data_array',
        '_datastr',
        '_conf_label_order',
        '_filters',
        '_structure_parser',
        'kwargs',
        ]

    def __init__(self, data=None, fasta=None, **kwargs):
        if data is not None:
            datastr = get_datastr(data)
        elif fasta is not None:
            datastr = fasta
        else:
            raise RuntimeError("Please specify either the data source or the FASTA sequence of the structure!")
        self._structure_parser = detect_structure_type(datastr)

        self._datastr = datastr
        self._conf_label_order = None
        self.kwargs = kwargs
        self.clear_filters()
        assert isinstance(self.filters, list)

    def __len__(self):
        return self.data_array.shape[0]

    def build(self):
        """
        Read structure raw data in :attr:`rawdata`.

        After `.build()`, filters and data can be accessed.
        """
        self._data_array = self._structure_parser(self._datastr, **self.kwargs)
        del self._datastr

    def clear_filters(self):
        """Clear/Deletes registered filters."""
        self._filters = []

    @property
    def data_array(self):
        """Contain structure data in the form of a Numpy array."""
        try:
            return self._data_array
        except AttributeError as err:
            errmsg = (
                'Please `.build()` the Structure before attempting to access'
                'its data'
                )
            raise NotBuiltError(errmsg=errmsg) from err

    @property
    def filters(self):
        """Filter functions registered ordered by registry record."""
        return self._filters

    @property
    def filtered_atoms(self):
        """
        Filter data array by the selected filters.

        Returns
        -------
        list
            The data in PDB format after filtering.
        """
        apply_filter = _APPLY_FILTER
        return np.array(list(reduce(
            apply_filter,
            self.filters,
            self.data_array,
            )))

    @property
    def chain_set(self):
        """All chain IDs present in the raw dataset."""  # noqa: D401
        return set(self.data_array[:, col_chainID])

    @property
    def coords(self):
        """
        Coordinates of the filtered atoms.

        As float.
        """
        return self.filtered_atoms[:, cols_coords].astype(np.float32)

    @coords.setter
    def coords(self, coords):
        self.data_array[:, cols_coords] = \
            np.round(coords, decimals=3).astype('<U8')



    @property
    def consecutive_residues(self):
        """Consecutive residue groups from filtered atoms."""
        # the structure.consecutive_residues
        # this will ignore the iCode
        # please use pdb-tool pdb_delinsertions before this step
        # PDBs downloaded with IDPConfGen already correct for these
        # see libs.libparse.delete_insertions
        return group_runs(self.residues)

    @property
    def fasta(self):
        """
        FASTA sequence of the :attr:`filtered_atoms` lines.

        HETATM residues with non-canonical codes are represented as X.
        """
        c, rs, rn = col_chainID, col_resSeq, col_resName

        chains = defaultdict(dict)
        for row in self.filtered_atoms:
            chains[row[c]].setdefault(row[rs], aa3to1.get(row[rn], 'X'))

        return {
            chain: ''.join(residues.values())
            for chain, residues in chains.items()
            }

    @property
    def residue_types(self):
        c, rs, rn = col_chainID, col_resSeq, col_resName

        chains = defaultdict(dict)
        for row in self.filtered_atoms:
            chains[row[c]].setdefault(row[rs], row[rn])
        return list(list(chains.values())[0].values())

    @property
    def filtered_residues(self):
        """Filter residues according to :attr:`filters`."""
        FA = self.filtered_atoms
        return [int(i) for i in dict.fromkeys(FA[:, col_resSeq])]

    @property
    def residues(self):
        """
        Residues of the structure.

        Without filtering, without chain separation.
        """
        return [int(i) for i in dict.fromkeys(self.data_array[:, col_resSeq])]

    @property
    def atom_labels(self):
        return self.data_array[:, col_name]

    @property
    def res_nums(self):
        return self.data_array[:, col_resSeq].astype(int)

    @property
    def res_labels(self):
        return self.data_array[:, col_resName]

    @residues.setter
    def residues(self, residue_idx):
        """
        Set the residue id of the current structure to a given value.

        Supposed to be applied only to a structure containing only one residue (i.e. sidechain template), 
        but does not perform this explicit check.
        """
        self.data_array[:, col_resSeq] = str(residue_idx)

    def pop_last_filter(self):
        """Pop last filter."""
        self._filters.pop()

    def add_filter(self, function):
        """Add a function as filter."""
        self.filters.append(function)

    def add_filter_record_name(self, record_name):
        """Add filter for record names."""
        self.filters.append(
            lambda x: x[col_record].startswith(record_name)
            )

    def add_filter_chain(self, chain):
        """Add filters for chain."""
        self.filters.append(lambda x: x[col_chainID] == chain)

    def add_filter_backbone(self, minimal=False):
        """Add filter to consider only backbone atoms."""
        ib = is_backbone
        self.filters.append(
            lambda x: ib(x[col_name], x[col_element], minimal=minimal)
            )

    def add_filter_resnum(self, resnum):
        """Add filters for residue number"""
        if type(resnum) is not str:
            resnum = str(resnum)
        self.filters.append(lambda x: x[col_resSeq] == resnum)

    def get_PDB(self, pdb_filters=None, renumber=True):
        """
        Convert Structure to PDB format.

        Considers only filtered lines.

        Returns
        -------
        generator
        """
        def _(i, f):
            return f(i)

        fs = self.filtered_atoms

        # renumber atoms
        if renumber:
            try:
                fs[:, col_serial] = np.arange(1, fs.shape[0] + 1).astype('<U8')
            except IndexError as err:
                errmsg = (
                    'Could not renumber atoms, most likely, because '
                    'there are no lines in selection.'
                    )
                err2 = EmptyFilterError(errmsg)
                raise err2 from err

        pdb_filters = pdb_filters or []

        lines = list(reduce(_, pdb_filters, structure_to_pdb(fs)))

        return lines

    def get_all_backbone_atom_coords(self):
        """
        Generate a copy of the backbone coords according to the order:
        ["N", "CA", "C", "O", "H1", "H2", "H3"] for N-terminal residue,
        ["N", "CA", "C", "O", "H"] for middle residues, and 
        then ["N", "CA", "C", "O", "H", "OXT"] for C-terminal residue

        For any proline residue, there will be no H (When PRO is the N-terminal
        there will still be H1 and H2)
        """
        pro_indicator = self.residue_types == "PRO"
        N_term_pro = pro_indicator[0]
        n_pro = np.sum(pro_indicator)
        num_backbone_atoms = len(self.residue_types) * 5 - n_pro + 3 # n_pro do not have hydrogen, additionally have H2, H3 and OXT
        backbone_coords = np.zeros((num_backbone_atoms, 3), dtype=np.float32)
        atoms = self.data_array
        coords = atoms[:, cols_coords]
        # First create coordinate lists for N, CA, C, O and H
        # For H, those positions corrisponding to prolines will be zero
        N_coords = coords[atoms[:, col_name] == 'N']
        CA_coords = coords[atoms[:, col_name] == 'CA']
        C_coords = coords[atoms[:, col_name] == 'C']
        O_coords = coords[atoms[:, col_name] == 'O']
        H_coords = np.zeros_like(O_coords)
        H_indices = np.where(np.logical_not(pro_indicator))[0]
        if not N_term_pro:
            H_indices = np.delete(H_indices, 0)
        H_coords[H_indices] = coords[atoms[:, col_name] == 'H']
        # Write in the N-terminal backbone coordinates
        backbone_coords[0] = N_coords[0]
        backbone_coords[1] = CA_coords[0]
        backbone_coords[2] = C_coords[0]
        backbone_coords[3] = O_coords[0]
        backbone_coords[4] = coords[atoms[:, col_name] == 'H1'][0]
        backbone_coords[5] = coords[atoms[:, col_name] == 'H2'][0]
        fillin_idx = 6
        if not N_term_pro:
            backbone_coords[6] = coords[atoms[:, col_name] == 'H3'][0]
            fillin_idx = 7
        # Non N-terminal residues
        for i in range(1, len(self.residue_types)):
            backbone_coords[fillin_idx] = N_coords[i]
            backbone_coords[fillin_idx + 1] = CA_coords[i]
            backbone_coords[fillin_idx + 2] = C_coords[i]
            backbone_coords[fillin_idx + 3] = O_coords[i]
            fillin_idx += 4
            if not pro_indicator[i]:
                backbone_coords[fillin_idx] = H_coords[i]
                fillin_idx += 1
        # Finally the OXT
        backbone_coords[fillin_idx] = coords[atoms[:, col_name] == 'OXT'][0]
        assert fillin_idx + 1 == num_backbone_atoms
        return backbone_coords

    def get_sorted_minimal_backbone_coords(self, filtered=False, with_O=False):
        """
        Generate a copy of the backbone coords sorted.

        When with_O is set to True, sorting according N, CA, C, O.
        
        When all_backbone is set to True, the order will be 
        

        Otherwise sorting according to N, CA, C.

        This method was created because some PDBs may not have the
        backbone atoms sorted properly.

        Parameters
        ----------
        filtered : bool, optional
            Whether consider current filters or raw data.
        """
        atoms = self.filtered_atoms if filtered else self.data_array
        coords = atoms[:, cols_coords]

        N_coords = coords[atoms[:, col_name] == 'N']
        CA_coords = coords[atoms[:, col_name] == 'CA']
        C_coords = coords[atoms[:, col_name] == 'C']

        N_num = N_coords.shape[0]
        CA_num = CA_coords.shape[0]
        C_num = C_coords.shape[0]

        if with_O:
            O_coords = coords[atoms[:, col_name] == 'O']
            O_num = O_coords.shape[0]

            num_backbone_atoms = sum([N_num, CA_num, C_num, O_num])
            assert num_backbone_atoms / 4 == N_num

            minimal_backbone = np.zeros((num_backbone_atoms, 3), dtype=np.float32)
            minimal_backbone[0:-3:4] = N_coords
            minimal_backbone[1:-2:4] = CA_coords
            minimal_backbone[2:-1:4] = C_coords
            minimal_backbone[3::4] = O_coords

        else:
            num_backbone_atoms = sum([N_num, CA_num, C_num])
            assert num_backbone_atoms / 3 == N_num

            minimal_backbone = np.zeros((num_backbone_atoms, 3), dtype=np.float32)
            minimal_backbone[0:-2:3] = N_coords
            minimal_backbone[1:-1:3] = CA_coords
            minimal_backbone[2::3] = C_coords

        return minimal_backbone

    def check_backbone_atom_completeness(self):
        '''
        Run check of backbone atom completeness and return a list containing all missing atoms from expected backbone atom list
        '''
        all_residue_atoms = {}
        for atom_label, res_num, res_label in zip(self.atom_labels, self.res_nums, self.res_labels):
            if res_num not in all_residue_atoms:
                all_residue_atoms[res_num] = {"label": res_label, "atoms": [atom_label]}
            else:
                all_residue_atoms[res_num]["atoms"].append(atom_label)
        n_term_idx = min(all_residue_atoms)
        c_term_idx = max(all_residue_atoms)
        missing_atoms = []
        for idx in all_residue_atoms:
            if idx == n_term_idx:
                expected_atoms = ["N", "CA", "C", "O", "H1", "H2"]
                if all_residue_atoms[idx]["label"] not in ["PRO", "HYP"]:
                    expected_atoms.append("H3")
            else:
                expected_atoms = ["N", "CA", "C", "O"]
                if all_residue_atoms[idx]["label"] not in ["PRO", "HYP"]:
                    expected_atoms.append("H")
                if idx == c_term_idx:
                    expected_atoms.append("OXT")
            residue_missing_atom = [item for item in expected_atoms \
                if item not in all_residue_atoms[idx]["atoms"]]
            missing_atoms.extend([(idx, item) for item in residue_missing_atom])
        return missing_atoms
        

    def remove_side_chains(self, retain_idxs=[]):
        """
        Create a copy of the current structure that removed all atoms beyond CB to be regrown by the MCSCE algorithm, except ones defined in resids to be retained.
        """
        copied_structure = deepcopy(self)
        retained_atoms_filter = np.array([atom in backbone_atoms for atom in copied_structure.data_array[:, col_name]])
        extra_pro_H_filter = (copied_structure.data_array[:, col_name] == 'H') & (copied_structure.data_array[:, col_resName] == 'PRO')
        if len(retain_idxs) > 0:
            for resid in retain_idxs:
                retained_atoms_filter = retained_atoms_filter | (copied_structure.data_array[:, col_resSeq] == str(resid))
        retained_atoms_filter = retained_atoms_filter & (~extra_pro_H_filter)
        copied_structure._data_array = copied_structure.data_array[retained_atoms_filter]
        return copied_structure #, None if np.all(retained_atoms_filter) else retained_atoms_filter

    def add_side_chain(self, res_idx, sidechain_template):
        template_structure, sc_atoms = sidechain_template
        self.add_filter_resnum(res_idx)
        N_CA_C_coords = self.get_sorted_minimal_backbone_coords(filtered=True)
        sc_all_atom_coords = place_sidechain_template(N_CA_C_coords, template_structure.coords)
        sidechain_data_arr = template_structure.data_array.copy()
        sidechain_data_arr[:, cols_coords] = np.round(sc_all_atom_coords, decimals=3).astype('<U8')
        sidechain_data_arr[:, col_resSeq] = str(res_idx)
        # conform to backbone residue labels but conform to sidechain records
        res_mask = (self.data_array[:, col_resSeq].astype(int) == res_idx)
        self.data_array[res_mask, col_record] = sidechain_data_arr[0, col_record]
        sidechain_data_arr[:, col_segid] = str(self.filtered_atoms[0, col_segid])
        sidechain_data_arr[:, col_chainID] = str(self.filtered_atoms[0, col_chainID])
        self.pop_last_filter()
        self._data_array = np.concatenate([self.data_array, sidechain_data_arr[sc_atoms]])



    def get_coords_in_conf_label_order(self, conf_label=None):
        if self._conf_label_order is not None:
            return self.coords[self._conf_label_order]
        else:
            if conf_label is None:
                raise RuntimeError("Coordinates in conformation label order has not yet been calculated!")
            if len(self.coords) != len(conf_label.atom_labels):
                raise RuntimeError("Structure and conformation label have different number of atoms. Please check!")
            order = []
            # Compare each resnum-resname-atomname combination to the lines in the data_array to decide the order of the atoms
            for label in zip(conf_label.res_nums, conf_label.res_labels, conf_label.atom_labels):
                match_query = (self.data_array[:, cols_labels] == np.array(label)).all(axis=-1)
                matched_pos = match_query.argmax()
                assert match_query[matched_pos]
                order.append(matched_pos)
            self._conf_label_order = order
            return self.coords[order]


    def write_PDB(self, filename, **kwargs):
        """Write Structure to PDB file."""
        lines = self.get_PDB(**kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                write_PDB(lines, filename)
            except UserWarning:
                raise EmptyFilterError(filename)

    def reorder(self, new_indices):
        """Reorder atoms acoording to the specified new indices, which is a list of the old indices in the new order"""
        self._data_array = self.data_array[new_indices]
        self.data_array[:, col_serial] = (np.arange(len(new_indices)) + 1).astype('<U8')

    def reorder_with_resnum(self):
        """Reorder all atoms in the Structure so that it strictly follows the residue number order"""
        sorted_indices = np.argsort(self.res_nums, kind="mergesort")
        self.reorder(sorted_indices)

    def reorder_by_atom_labels(self, atom_labels):
        """Reorder all atoms according to the atomic order defined in atom_labels"""
        reordered_indices = []
        for ri in range(1, len(self.residues) + 1):
            residue_indices = np.where(atom_labels.res_nums == ri)[0]
            assert (self.res_nums[residue_indices] == ri).all()
            start_idx = residue_indices[0]
            old_atom_labels = list(self.atom_labels[residue_indices])
            for new_atom_name in atom_labels.atom_labels[residue_indices]:
                reordered_indices.append(start_idx + old_atom_labels.index(new_atom_name))
        self.reorder(reordered_indices)



def parse_pdb_to_array(datastr, which='both'):
    """
    Transform PDB data into an array.

    Parameters
    ----------
    datastr : str
        String representing the PDB format v3 file.

    which : str
        Which lines to consider ['ATOM', 'HETATM', 'both'].
        Defaults to `'both'`, considers both 'ATOM' and 'HETATM'.

    Returns
    -------
    numpy.ndarray of (N, len(`libpdb.atom_slicers`))
        Where N are the number of ATOM and/or HETATM lines,
        and axis=1 the number of fields in ATOM/HETATM lines according
        to the PDB format v3.
    """
    # require
    assert isinstance(datastr, str), \
        f'`datastr` is not str: {type(datastr)} instead'

    model_idx = RE_MODEL.search(datastr)
    endmdl_idx = RE_ENDMDL.search(datastr)

    if bool(model_idx) + bool(endmdl_idx) == 1:
        # only one is True
        raise PDBFormatError('Found MODEL and not ENDMDL, or vice-versa')

    if model_idx and endmdl_idx:
        start = model_idx.span()[1]
        end = endmdl_idx.span()[0] - 1
        lines = datastr[start: end].split('\n')

    else:
        lines = datastr.split('\n')

    record_lines = filter_record_lines(lines, which=which)
    data_array = gen_empty_structure_data_array(len(record_lines))
    populate_structure_array_from_pdb(record_lines, data_array)
    return data_array


def parse_fasta_to_array(datastr, **kwargs):
    """
    Establish backbone data array from the specified FASTA sequence
    """
    n_residues = len(datastr)
    residues_aa3 = translate_seq_to_3l(datastr, histidine_protonation='HIP')
    # TODO: test ptm, comment out
    #if 'HIP' in residues_aa3: residues_aa3[residues_aa3.index('HIP')] = 'H2D'
   
    # For each residue we have N, CA, C, O, HN, then there are two added HN on N-terminal and one OXT on C terminal
    # Also prolines do not have HN
    data_array = gen_empty_structure_data_array(5 * n_residues + 3 - datastr.count('P')) 
    populate_structure_array_with_backbone(residues_aa3, data_array)
    return data_array

def parse_cif_to_array(datastr, **kwargs):
    """
    Parse mmCIF protein data to array.

    Array is as given by :func:`gen_empty_structure_data_array`.
    """
    cif = CIFParser(datastr, **kwargs)
    number_of_atoms = len(cif)
    data_array = gen_empty_structure_data_array(number_of_atoms)

    for ii in range(number_of_atoms):
        data_array[ii, :] = cif.get_line_elements_for_PDB(line=ii)

    return data_array


def gen_empty_structure_data_array(number_of_atoms):
    """
    Generate an array data structure to contain structure data.

    Parameters
    ----------
    number_of_atoms : int
        The number of atoms in the structure.
        Determines the size of the axis 0 of the structure array.

    Returns
    -------
    np.ndarray of (N, :attr:`libpdb.atom_slicers), dtype = '<U8'
        Where N is the ``number_of_atoms``.
    """
    # require
    assert isinstance(number_of_atoms, int), \
        f'`number_of_atoms` is not int, {type(number_of_atoms)} '
    assert number_of_atoms > 0, \
        f'or number is less than zero: {number_of_atoms}.'

    return np.empty(
        (number_of_atoms, len(libpdb.atom_slicers)),
        dtype='<U8',
        )


def generate_residue_labels(*residue_labels, fmt=None, delimiter=' - '):
    """
    Generate residue labels column.

    Concatenate labels in `residue_labels` using
        `concatenate_residue_labels`.

    Parameters
    ----------
    fmt : str, optional
        The string formatter by default we consider backbone atoms
        of a protein with less than 1000 residues.
        Defaults to `None`, uses '{:<8}', 8 or multiple of 8 according
        to length of residue_labels.
    """
    if not fmt:
        # 11 because 8 + len(' - ')
        fmt = '{:<' + str(len(residue_labels) * (8 + len(delimiter))) + '}'

    concat = (
        concatenate_residue_labels(label_tuple)
        for label_tuple in residue_labels
        )
    return [fmt.format(delimiter.join(clabels)) for clabels in zip(*concat)]


def generate_backbone_pairs_labels(da):
    """
    Generate backbone atom pairs labels.

    Used to create columns in report summaries.

    Parameters
    ----------
    da : Structure.data_array - like

    Returns
    -------
    Numpy Array of dtype str, shape (N,)
        Where N is the number of minimal backbone atoms.

    """
    # masks from minimal backbone atoms
    N_mask = da[:, col_name] == 'N'
    CA_mask = da[:, col_name] == 'CA'
    C_mask = da[:, col_name] == 'C'

    # prepares labels column format
    # number of digit of the highest residue
    # 5 is 3 from 3-letter aa code and 2 from 'CA' label length
    max_len = len(str(np.sum(N_mask))) + 5
    fmt_res = '{:<' + str(max_len) + '}'

    # Generate label pairs
    N_CA_labels = generate_residue_labels(
        da[:, cols_labels][N_mask],
        da[:, cols_labels][CA_mask],
        fmt=fmt_res,
        )

    CA_C_labels = generate_residue_labels(
        da[:, cols_labels][CA_mask],
        da[:, cols_labels][C_mask],
        fmt=fmt_res,
        )

    C_N_labels = generate_residue_labels(
        da[:, cols_labels][C_mask][:-1],
        da[:, cols_labels][N_mask][1:],
        fmt=fmt_res,
        )

    # prepares the labels array
    # max_label_len to use the exact needed memory size
    _ = [N_CA_labels, CA_C_labels, C_N_labels]
    number_of_bb_atoms = sum(len(i) for i in _)
    max_label_len = max(len(i) for i in N_CA_labels)
    labels = np.zeros(number_of_bb_atoms, dtype=f'<U{max_label_len}')
    labels[0::3] = N_CA_labels
    labels[1::3] = CA_C_labels
    labels[2::3] = C_N_labels

    return labels


def concatenate_residue_labels(labels):
    """
    Concatenate residue labels.

    This function is a generator.

    Parameters
    ----------
    labels : numpy array of shape (N, M)
        Where N is the number of rows, and M the number of columns
        with the labels to be concatenated.
    """
    empty_join = ''.join
    return (empty_join(res_label) for res_label in labels)


def populate_structure_array_from_pdb(record_lines, data_array):
    """
    Populate structure array from PDB lines.

    Parameters
    ----------
    record_lines : list-like
        The PDB record lines (ATOM or HETATM) to parse.

    data_array : np.ndarray
        The array to populate.

    Returns
    -------
    None
        Populates array in place.
    """
    AS = libpdb.atom_slicers
    for row, line in enumerate(record_lines):
        for column, slicer_item in enumerate(AS):
            data_array[row, column] = line[slicer_item].strip()
            if column == 2:
                # make sure the pdb is using Amber naming: multiple H counting is at the end
                original_name = data_array[row, column]
                if original_name[0].isdigit():
                    assert "H" in original_name
                    new_name = original_name[1:] + original_name[0]
                    data_array[row, column] = new_name

def populate_structure_array_with_backbone(residue_codes, data_array):
    """
    Populate structure array by specifying the backbone atoms with coordinates default to all zeros
    """
    row_num = 0
    for residx, rescode in enumerate(residue_codes):
        resid = residx + 1  # resid shoud start from 1
        if resid == 1: # N terminal
            if rescode != 'PRO':
                fill_in_atoms = ["N", "CA", "C", "O", "H1", "H2", "H3"]
            else:
                fill_in_atoms = ["N", "CA", "C", "O", "H1", "H2"]
        else:
            # Proline middle residue
            fill_in_atoms = ["N", "CA", "C", "O"]
            if rescode != 'PRO': # Non-proline middle residue
                fill_in_atoms.append("H")
            if resid == len(residue_codes): # C terminal
                fill_in_atoms.append("OXT")

        for atom in fill_in_atoms:
            data_array[row_num, col_record] = "ATOM"
            data_array[row_num, col_serial] = str(row_num + 1)
            data_array[row_num, col_name] = atom
            data_array[row_num, col_resName] = rescode
            data_array[row_num, col_chainID] = "A"
            data_array[row_num, col_resSeq] = str(resid)
            data_array[row_num, col_x] = "0.000"
            data_array[row_num, col_y] = "0.000"
            data_array[row_num, col_z] = "0.000"
            data_array[row_num, col_occ] = "1.00"
            data_array[row_num, col_temp] = "0.00"
            data_array[row_num, col_element] = atom[0]
            row_num += 1


def filter_record_lines(lines, which='both'):
    """Filter lines to get record lines only."""
    RH = record_line_headings
    try:
        return [line for line in lines if line.startswith(RH[which])]
    except KeyError as err:
        err2 = ValueError(f'`which` got an unexpected value \'{which}\'.')
        raise err2 from err


def get_datastr(data):
    """
    Get data in string format.

    Can parse data from several formats:

    * Path, reads file content
    * bytes, converst to str
    * str, returns the input

    Returns
    -------
    str
        That represents the data
    """
    t2s = type2string
    data_type = type(data)
    try:
        datastr = t2s[data_type](data)
    except KeyError as err:
        err2 = NotImplementedError('Struture data not of proper type')
        raise err2 from err
    assert isinstance(datastr, str)
    return datastr


def detect_structure_type(datastr):
    """
    Detect Structure data parser.

    Uses `structure_parsers`.

    Returns
    -------
    func or class
        That which can parse `datastr` to a :class:`Structure'.
    """
    sp = structure_parsers
    for condition, parser in sp:
        if condition(datastr):
            return parser
    raise ParserNotFoundError


def write_PDB(lines, filename):
    """
    Write Structure data format to PDB.

    Parameters
    ----------
    lines : list or np.ndarray
        Lines contains PDB data as according to `parse_pdb_to_array`.

    filename : str or Path
        The name of the output PDB file.
    """
    # use join here because lines can be a generator
    concat_lines = '\n'.join(lines)
    if concat_lines:
        with open(filename, 'w') as fh:
            fh.write(concat_lines)
            fh.write('\n')
    else:
        warnings.warn('Empty lines, nothing to write, ignoring.', UserWarning)


def structure_to_pdb(atoms):
    """
    Convert table to PDB formatted lines.

    Parameters
    ----------
    atoms : np.ndarray, shape (N, 16) or similar data structure
        Where N is the number of atoms and 16 the number of cols.

    Yields
    ------
    Formatted PDB line according to `libpdb.atom_line_formatter`.
    """
    for line in atoms:
        values = [func(i) for i, func in zip(line, libpdb.atom_format_funcs)]
        values[col_name] = libpdb.format_atom_name(
            values[col_name],
            values[col_element],
            )
        yield libpdb.atom_line_formatter.format(*values)


col_record = 0
col_serial = 1
col_name = 2  # atom name
col_altLoc = 3
col_resName = 4
col_chainID = 5
col_resSeq = 6
col_iCode = 7
col_x = 8
col_y = 9
col_z = 10
col_occ = 11
col_temp = 12
col_segid = 13
col_element = 14
col_model = 15


cols_coords_slice = slice(8, 11)
cols_coords = [col_x, col_y, col_z]
cols_labels = [col_resSeq, col_resName, col_name]


# this servers read_pdb_data_to_array mainly
# it is here for performance
record_line_headings = {
    'both': ('ATOM', 'HETATM'),
    'ATOM': 'ATOM',
    'HETATM': 'HETATM',
    }


# order matters
structure_parsers = [
    (is_cif, parse_cif_to_array),
    (libpdb.is_pdb, parse_pdb_to_array),
    (is_valid_fasta, parse_fasta_to_array)
    ]


def _APPLY_FILTER(it, func):
    return filter(func, it)


def is_backbone(atom, element, minimal=False):
    """
    Whether `atom` is a protein backbone atom or not.

    Parameters
    ----------
    atom : str
        The atom name.

    element : str
        The element name.

    minimal : bool
        If `True` considers only `C` and `N` elements.
        `False`, considers also `O`.
    """
    e = element.strip()
    a = atom.strip()
    elements = {
        True: ('N', 'C'),
        False: ('N', 'C', 'O'),
        }
    # elements is needed because of atoms in HETATM entries
    # for example 'CA' is calcium
    return a in ('N', 'CA', 'C', 'O') and e in elements[minimal]


def save_structure_by_chains(
        pdb_data,
        pdbname,
        altlocs=('A', '', ' ', '1'),   # CIF: 6uwi chain D has altloc 1
        chains=None,
        record_name=('ATOM', 'HETATM'),
        renumber=True,
        **kwargs,
        ):
    """
    Save PDBs/mmCIF in separated chains (PDB format).

    Logic to parse PDBs from RCSB.
    """
    # local assignments for speed boost :D
    _AE = _allowed_elements
    _S = Structure
    _DI = [delete_insertions]

    pdbdata = _S(pdb_data)
    pdbdata.build()

    chain_set = pdbdata.chain_set

    chains = chains or chain_set

    add_filter = pdbdata.add_filter
    pdbdata.add_filter_record_name(record_name)
    add_filter(lambda x: x[col_element] in _AE)
    add_filter(lambda x: x[col_altLoc] in altlocs)

    for chain in chains:

        # writes chains always in upper case because chain IDs given by
        # Dunbrack lab are always in upper case letters
        chaincode = f'{pdbname}_{chain}'

        # this operation can't be performed before because
        # until here there is not way to assure if the chain being
        # downloaded is actualy in the blocked_ids.
        # because at the CLI level the user can add only the PDBID
        # to indicate download all chains, while some may be restricted

        # upper and lower case combinations:
        possible_cases = sample_case(chain)
        # cases that exist in the structure
        cases_that_actually_exist = chain_set.intersection(possible_cases)
        # this trick places `chain` first in the for loop because
        # it has the highest probability to be the one required
        cases_that_actually_exist.discard(chain)
        probe_cases = [chain] + list(cases_that_actually_exist)

        for chain_case in probe_cases:

            pdbdata.add_filter_chain(chain_case)
            fout = f'{chaincode}.pdb'

            try:
                pdb_lines = pdbdata.get_PDB(pdb_filters=_DI)
            except EmptyFilterError as err:
                err2 = \
                    EmptyFilterError(f'for chain {pdbname}_{chain_case}')
                errlog = (
                    f'{repr(err)}\n'
                    f'{repr(err2)}\n'
                    f'{traceback.format_exc()}\n'
                    'continuing to new chain\n'
                    )
                log.debug(errlog)
                continue
            else:
                if all(line.startswith('HETATM') for line in pdb_lines):
                    log.debug(
                        f'Found only HETATM for {chain_case}, '
                        'continuing with next chain.'
                        )
                    continue
                yield fout, '\n'.join(pdb_lines)
                break
            finally:
                pdbdata.pop_last_filter()
        else:
            log.debug(f'Failed to download {chaincode}')
