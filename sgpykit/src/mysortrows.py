import numpy as np


def mysortrows_stable(A, tol=1e-14): #, i=None, i_ord=None, n=None):
    """
    Similar to Matlab builtin function sortrows. Given a matrix A
    of real numbers, sorts the rows in lexicographic order; entries
    that differ less than Tol are treated as equal (default Tol is 1e-14).

    Parameters
    ----------
    A : array_like
        Input matrix to be sorted.
    tol : float, optional
        Tolerance used to identify coincident entries (default 1e-14).

    Returns
    -------
    B : ndarray
        Sorted matrix.
    idx : ndarray
        Index vector such that A[idx,:] = B.
    """
    # usage:
    #   [B,i]=mysortrows(A,Tol)
    #   input
    #     A: input matrix
    #     Tol: (optional) tolerance used to identify coincident entries
    #          (default 1e-14)
    #   output
    #     B: sorted matrix
    #     i: index vector such that A[i,:]=B

    ## REIMPLEMENTATION for mysortrows
    a = np.atleast_2d(A)
    rounded_matrix = np.round(a / tol) * tol
    idx = np.lexsort(rounded_matrix.T[::-1]) # 0-based idx
    B = rounded_matrix[idx]
    return B, idx


def mysortrows(A, tol=1e-14):
    """
    Sort the rows of a matrix in lexicographic order with a tolerance.

    Given a matrix A of real numbers, sorts the rows in lexicographic order.
    Entries that differ less than Tol are treated as equal (default Tol is 1e-14).

    Parameters
    ----------
    A : array_like
        Input matrix to be sorted.
    tol : float, optional
        Tolerance used to identify coincident entries (default 1e-14).

    Returns
    -------
    B : ndarray
        Sorted matrix.
    idx : ndarray
        Index vector such that A[idx,:] = B.
    """
    a = np.atleast_2d(A)
    n_rows, n_cols = a.shape

    # Handle empty matrix
    if n_rows == 0:
        return a, np.array([], dtype=np.int64)

    # Start with one block containing all row indices (0-based)
    blocks = [list(range(n_rows))]

    # Process each column left-to-right
    for col in range(n_cols):
        new_blocks = []
        for block in blocks:
            if len(block) <= 1:
                new_blocks.append(block)
                continue

            # Sort block by current column (STABLE SORT)
            block_sorted = sorted(block, key=lambda i: a[i, col])

            # Split into sub-blocks where consecutive rows are within Tol
            sub_blocks = []
            current = [block_sorted[0]]
            for j in range(1, len(block_sorted)):
                diff = abs(a[block_sorted[j], col] - a[block_sorted[j - 1], col])
                if diff <= tol:  # Group if within tolerance
                    current.append(block_sorted[j])
                else:
                    sub_blocks.append(current)
                    current = [block_sorted[j]]
            sub_blocks.append(current)
            new_blocks.extend(sub_blocks)

        blocks = new_blocks  # Proceed to next column with new blocks

    # Flatten blocks to get sorted row indices
    sorted_indices = [idx for block in blocks for idx in block]
    B = a[sorted_indices, :]
    idx = np.array(sorted_indices)
    return B, idx
