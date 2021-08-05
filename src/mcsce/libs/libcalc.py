"""
Several helper functions for calculating pairwise quantities.

Original code in this file from IDP Conformer Generator package
(https://github.com/julie-forman-kay-lab/IDPConformerGenerator)
developed by Joao M. C. Teixeira (@joaomcteixeira), and added to the
MSCCE repository in commit 30e417937968f3c6ef09d8c06a22d54792297161.
"""
import numpy as np
from numba import njit

@njit
def calc_all_vs_all_dists(coords):
    """
    Calculate the upper half of all vs. all distances.

    Reproduces the operations of scipy.spatial.distance.pdist.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3), dtype=np.float64

    Returns
    -------
    np.ndarray, shape ((N * N - N) // 2,), dytpe=np.float64
    """
    len_ = coords.shape[0]
    shape = ((len_ * len_ - len_) // 2,)
    results = np.empty(shape, dtype=np.float64)

    c = 1
    i = 0
    for a in coords:
        for b in coords[c:]:
            x = b[0] - a[0]
            y = b[1] - a[1]
            z = b[2] - a[2]
            results[i] = (x * x + y * y + z * z) ** 0.5
            i += 1
        c += 1

    return results

@njit
def sum_upper_diagonal_raw(data, result):
    """
    Calculate outer sum for upper diagonal with for loops.

    The use of for-loop based calculation avoids the creation of very
    large arrays using numpy outer derivates. This function is thought
    to be jut compiled.

    Does not create new data structure. It requires the output structure
    to be provided. Hence, modifies in place. This was decided so
    because this function is thought to be jit compiled and errors with
    the creation of very large arrays were rising. By passing the output
    array as a function argument, errors related to memory freeing are
    avoided.

    Parameters
    ----------
    data : an interable of Numbers, of length N

    result : a mutable sequence, either list of np.ndarray,
             of length N*(N-1)//2
    """
    c = 0
    len_ = len(data)
    for i in range(len_ - 1):
        for j in range(i + 1, len_):
            result[c] = data[i] + data[j]
            c += 1

    # assert result.size == (data.size * data.size - data.size) // 2
    # assert abs(result[0] - (data[0] + data[1])) < 0.0000001
    # assert abs(result[-1] - (data[-2] + data[-1])) < 0.0000001
    return

@njit
def multiply_upper_diagonal_raw(data, result):
    """
    Calculate the upper diagonal multiplication with for loops.

    The use of for-loop based calculation avoids the creation of very
    large arrays using numpy outer derivatives. This function is thought
    to be njit compiled.

    Does not create new data structure. It requires the output structure
    to be provided. Hence, modifies in place. This was decided so
    because this function is thought to be jit compiled and errors with
    the creation of very large arrays were rising. By passing the output
    array as a function argument, errors related to memory freeing are
    avoided.

    Parameters
    ----------
    data : an interable of Numbers, of length N

    result : a mutable sequence, either list of np.ndarray,
             of length N*(N-1)//2
    """
    c = 0
    len_ = len(data)
    for i in range(len_ - 1):
        for j in range(i + 1, len_):
            result[c] = data[i] * data[j]
            c += 1


@njit
def calc_angle_coords(
        coords,
        ARCCOS=np.arccos,
        DOT=np.dot,
        NORM=np.linalg.norm,
        ):
    """Calculate the angle between two vectors."""
    # https://stackoverflow.com/questions/2827393/
    v1 = coords[0] - coords[1]
    v2 = coords[2] - coords[1]
    return calc_angle(v1, v2)


@njit
def calc_angle(
        v1, v2,
        ARCCOS=np.arccos,
        DOT=np.dot,
        NORM=np.linalg.norm,
        ):
    """Calculate the angle between two vectors."""
    # https://stackoverflow.com/questions/2827393/
    v1_u = v1 / NORM(v1)
    v2_u = v2 / NORM(v2)

    dot_ncan = np.dot(v1_u, v2_u)

    if dot_ncan < -1.0:
        dot_ncan_clean = -1.0

    elif dot_ncan > 1.0:
        dot_ncan_clean = 1.0

    else:
        dot_ncan_clean = dot_ncan

    return ARCCOS(dot_ncan_clean)

# @njit
def calc_torsion_angles(
        coords,
        ARCTAN2=np.arctan2,
        CROSS=np.cross,
        DIAGONAL=np.diagonal,
        MATMUL=np.matmul,
        NORM=np.linalg.norm,
        ):
    """
    Calculate torsion angles from sequential coordinates.

    Uses ``NumPy`` to compute angles in a vectorized fashion.
    Sign of the torsion angle is also calculated.

    Uses Prof. Azevedo implementation:
    https://azevedolab.net/resources/dihedral_angle.pdf

    Example
    -------
    Given the sequential coords that represent a dummy molecule of
    four atoms:

    >>> xyz = numpy.array([
    >>>     [0.06360, -0.79573, 1.21644],
    >>>     [-0.47370, -0.10913, 0.77737],
    >>>     [-1.75288, -0.51877, 1.33236],
    >>>     [-2.29018, 0.16783, 0.89329],
    >>>     ])

    A1---A2
           \
            \
            A3---A4

    Calculates the torsion angle in A2-A3 that would place A4 in respect
    to the plane (A1, A2, A3).

    Likewise, for a chain of N atoms A1, ..., An, calculates the torsion
    angles in (A2, A3) to (An-2, An-1). (A1, A2) and (An-1, An) do not
    have torsion angles.

    If coords represent a protein backbone consisting of N, CA, and C
    atoms and starting at the N-terminal, the torsion angles are given
    by the following slices to the resulting array:

    - phi (N-CA), [2::3]
    - psi (CA-C), [::3]
    - omega (C-N), [1::3]

    Parameters
    ----------
    coords : numpy.ndarray of shape (N>=4, 3)
        Where `N` is the number of atoms, must be equal or above 4.

    Returns
    -------
    numpy.ndarray of shape (N - 3,)
        The torsion angles in radians.
        If you want to convert those to degrees just apply
        ``np.degrees`` to the returned result.
    """
    # requires
    assert coords.shape[0] > 3
    assert coords.shape[1] == 3

    crds = coords.T

    # Yes, I always write explicit array indices! :-)
    q_vecs = crds[:, 1:] - crds[:, :-1]
    cross = CROSS(q_vecs[:, :-1], q_vecs[:, 1:], axis=0)
    unitary = cross / NORM(cross, axis=0)

    # components
    # u0 comes handy to define because it fits u1
    u0 = unitary[:, :-1]

    # u1 is the unitary cross products of the second plane
    # that is the unitary q2xq3, obviously applied to the whole chain
    u1 = unitary[:, 1:]

    # u3 is the unitary of the bonds that have a torsion representation,
    # those are all but the first and the last
    u3 = q_vecs[:, 1:-1] / NORM(q_vecs[:, 1:-1], axis=0)

    # u2
    # there is no need to further select dimensions for u2, those have
    # been already sliced in u1 and u3.
    u2 = CROSS(u3, u1, axis=0)

    # calculating cos and sin of the torsion angle
    # here we need to use the .T and np.diagonal trick to achieve
    # broadcasting along the whole coords chain
    # np.matmul is preferred to np.dot in this case
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    cos_theta = DIAGONAL(MATMUL(u0.T, u1))
    sin_theta = DIAGONAL(MATMUL(u0.T, u2))

    # torsion angles
    return -ARCTAN2(sin_theta, cos_theta)

@njit
def hamiltonian_multiplication_Q(a1, b1, c1, d1, a2, b2, c2, d2):
    """Hamiltonian Multiplication."""
    return (
        (a1 * a2) - (b1 * b2) - (c1 * c2) - (d1 * d2),
        (a1 * b2) + (b1 * a2) + (c1 * d2) - (d1 * c2),
        (a1 * c2) - (b1 * d2) + (c1 * a2) + (d1 * b2),
        (a1 * d2) + (b1 * c2) - (c1 * b2) + (d1 * a2),
        )

@njit
def rotate_coordinates_Q(
        coords,
        rot_vec,
        angle_rad,
        ARRAY=np.array,
        HMQ=hamiltonian_multiplication_Q,
        VSTACK=np.vstack,
        ):
    """
    Rotate coordinates by radians along an axis.

    Rotates using quaternion operations.

    Parameters
    ----------
    coords : nd.array (N, 3), dtype=np.float64
        The coordinates to rotate.

    rot_vec : (,3)
        A 3D space vector around which to rotate coords.
        Rotation vector **must** be a unitary vector.

    angle_rad : float
        The angle in radians to rotate the coords.

    Returns
    -------
    nd.array shape (N, 3), dtype=np.float64
        The rotated coordinates
    """
    # assert coords.shape[1] == 3

    b2, b3, b4 = np.sin(angle_rad / 2) * rot_vec
    b1 = np.cos(angle_rad / 2)

    c1, c2, c3, c4 = HMQ(
        b1, b2, b3, b4,
        0, coords[:, 0], coords[:, 1], coords[:, 2],
        )

    _, d2, d3, d4 = HMQ(
        c1, c2, c3, c4,
        b1, -b2, -b3, -b4,
        )

    rotated = VSTACK((d2, d3, d4)).T

    assert rotated.shape[1] == 3
    return rotated

@njit
def place_sidechain_template(
        bb_cnf,
        ss_template,
        CROSS=np.cross,
        NORM=np.linalg.norm,
        ):
    """
    Place sidechain templates on backbone.

    Sidechain residue template is expected to have CA already at 0,0,0.

    Parameters
    ----------
    bb_cnf : numpy nd.array, shape (3, 3), dtype=float64
        The backbone coords in the form of: N-CA-C
        Coordinates are not expected to be at any particular position.

    ss_template : numpy nd.array, shape (M, 3), dtype=float64
        The sidechain all-atom template. **Expected** to have the CA atom
        at the origin (0, 0, 0), and the first 3 atoms are N, CA, C. This requirement could be easily
        removed but it is maintained for performance reasons and
        considering in the context where this function is meant
        to be used.

    Returns
    -------
    nd.array, shape (M, 3), dtype=float64
        The displaced side chain coords. All atoms are returned.
    """
    # places bb with CA at 0,0,0
    bbtmp = np.full(bb_cnf.shape, np.nan, dtype=np.float32)
    bbtmp[:, :] = bb_cnf[:, :] - bb_cnf[1, :]

    # the sidechain residue template is expected to have CA
    # already at the the origin (0,0,0)
    N_CA = bbtmp[0, :]
    N_CA_ = ss_template[0, :]

    N_CA_N = calc_angle(N_CA, N_CA_)

    # rotation vector
    rv = CROSS(N_CA_, N_CA)
    rvu = rv / NORM(rv)

    # aligns the N-CA vectors
    rot1 = rotate_coordinates_Q(ss_template, rvu, N_CA_N).astype(np.float32)

    # starts the second rotation to align the CA-C vectors
    # calculates the cross vectors of the planes N-CA-C
    cross_cnf = CROSS(bbtmp[0, :], bbtmp[2, :])
    cross_ss = CROSS(rot1[0, :], rot1[2, :])

    # the angle of rotation is the angle between the plane normal
    angle = calc_angle(cross_ss, cross_cnf)

    # plane rotation vector is the cross vector between the two plane normals
    rv = CROSS(cross_ss, cross_cnf)
    rvu = rv / NORM(rv)

    # aligns to the CA-C vector maintaining the N-CA in place
    rot2 = rotate_coordinates_Q(rot1, rvu, angle)

    return rot2[:, :] + bb_cnf[1, :]
