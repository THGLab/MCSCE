"""
Tools for conformer building operations.

Original code in this file from IDP Conformer Generator package
(https://github.com/julie-forman-kay-lab/IDPConformerGenerator)
developed by Joao M. C. Teixeira (@joaomcteixeira), and added to the
MSCCE repository in commit 30e417937968f3c6ef09d8c06a22d54792297161.
Modifications herein are of MCSCE authors.
"""


from collections import Counter, namedtuple
import numpy as np

from mcsce.libs.libparse import translate_seq_to_3l

ConfLabels = namedtuple(
    'ConfLabels',
    [
        'atom_labels',
        'res_nums',
        'res_labels',
        ]
    )
"""
Contain label information for a protein/conformer.

Named indexes
-------------
atom_labels
res_nums
res_labels
"""

def init_conflabels(*args, **kwargs):
    """
    Create atom and residue labels from sequence.

    Parameters
    ----------
    *args, **kwargs
        Whichever `:func:create_conformer_labels` accepts.

    Returns
    -------
    namedtuple
        ConfLabels named tuple populated according to input sequence.

    See Also
    --------
    create_conformer_labels()
    ConfLabels
    """
    return ConfLabels(*create_conformer_labels(*args, **kwargs))


def create_conformer_labels(
        input_seq,
        atom_names_definition,
        transfunc=translate_seq_to_3l,
        **kwargs
        ):
    """
    Create all atom/residue labels model based on an input sequence.

    The labels are those expected for a all atom model PDB file. Hence,
    residue labels are repeated as needed in order to exist one residue
    label/number per atom.

    Parameters
    ----------
    input_seq : str
        The protein input sequence in 1-letter code format.

    atom_names_definition : dict
        Keys are residue identity and values are list/tuple of strings
        identifying atoms. Atom names should be sorted by the desired
        order.

    transfunc : func
        Function used to translate 1-letter input sequence to 3-letter
        sequence code.

    Returns
    -------
    tuple (atom labels, residue numbers, residue labels)
        Each is a np.ndarray of types: '<U4', np.int, and '<U3' and
        shape (N,) where N is the number of atoms.
        The three arrays have the same length.
    """
    input_seq_3_letters = transfunc(input_seq, kwargs.get("histidine_protonation", "HIS"))
    # /
    # prepares data based on the input sequence
    # considers sidechain all-atoms
    atom_labels = np.array(
        make_list_atom_labels(
            input_seq,
            atom_names_definition,
            )
        )
    num_atoms = len(atom_labels)

    # /
    # per atom labels
    residue_numbers = np.empty(num_atoms, dtype=np.int)
    residue_labels = np.empty(num_atoms, dtype='<U3')

    # generators
    _res_nums_gen = gen_residue_number_per_atom(atom_labels, start=1)
    _res_labels_gen = \
        gen_3l_residue_labels_per_atom(input_seq_3_letters, atom_labels)

    # fills empty arrays from generators
    _zipit = zip(range(num_atoms), _res_nums_gen, _res_labels_gen)
    for _i, _num, _label in _zipit:
        residue_numbers[_i] = _num
        residue_labels[_i] = _label

    # maniatic cleaning from pre-function isolation
    del _res_labels_gen, _res_nums_gen, _zipit

    # ensure
    assert len(residue_numbers) == num_atoms
    assert len(residue_labels) == num_atoms, (len(residue_labels), num_atoms)
    # ?
    return atom_labels, residue_numbers, residue_labels

def make_list_atom_labels(input_seq, atom_labels_dictionary):
    """
    Make a list of the atom labels for an `input_seq`.

    Considers the N-terminal to be protonated H1 to H3.
    Adds also 'OXT' terminal label.

    Parameters
    ----------
    input_seq : str
        1-letter amino-acid sequence.

    atom_labels_dictionary : dict
        The ORDERED atom labels per residue.

    Returns
    -------
    list
        List of consecutive atom labels for the protein.
    """
    labels = []
    LE = labels.extend

    first_residue_atoms = atom_labels_dictionary[input_seq[0]]

    # the first residue is a special case, we add here the three protons
    # for consistency with the forcefield
    # TODO: parametrize? multiple forcefields?
    for atom in first_residue_atoms:
        if atom == 'H':
            LE(('H1', 'H2', 'H3'))
        else:
            labels.append(atom)

    for residue in input_seq[1:]:
        LE(atom_labels_dictionary[residue])

    labels.append('OXT')

    assert Counter(labels)['N'] == len(input_seq)
    assert labels[-1] == 'OXT'
    assert 'H1' in labels
    assert 'H2' in labels
    assert 'H3' in labels
    return labels

def gen_residue_number_per_atom(atom_labels, start=1):
    """
    Create a list of residue numbers based on atom labels.

    This is a contextualized function, not an abstracted one.
    Considers `N` to be the first atom of the residue.

    Yields
    ------
    ints
        The integer residue number per atom label.
    """
    assert atom_labels[0] == 'N', atom_labels[0]

    # creates a seamless interface between human and python 0-indexes
    start -= 1
    for al in atom_labels:
        if al == 'N':
            start += 1
        yield start


def gen_3l_residue_labels_per_atom(
        input_seq_3letter,
        atom_labels,
        ):
    """
    Generate residue 3-letter labels per atom.

    Parameters
    ----------
    input_seq_3letter : list of 3letter residue codes
        Most not be a generator.

    atom_labels : list or tuple of atom labels
        Most not be a generator.

    Yields
    ------
    String of length 3
         The 3-letter residue code per atom.
    """
    _count_labels = Counter(atom_labels)['N']
    _len = len(input_seq_3letter)
    assert _count_labels == _len, (_count_labels, _len)

    counter = -1
    for atom in atom_labels:
        if atom == 'N':
            counter += 1
        yield input_seq_3letter[counter]
