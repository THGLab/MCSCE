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