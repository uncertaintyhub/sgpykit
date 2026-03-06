import logging
import numpy as np

from sgpykit.tools.idxset_functions.check_index_admissibility import check_index_admissibility
from sgpykit.util import matlab, misc
from sgpykit.util.checks import is_number_or_list_of_numbers_nonnegative
from sgpykit.util.misc import matlab_to_python_index, python_to_matlab_index

logger = logging.getLogger(__name__)


def check_set_admissibility(I, matlab_idx=True):
    """
    Check if a given index set is admissible and return the admissible set.

    Parameters
    ----------
    I : array_like
        The index set to check for admissibility.
    matlab_idx : bool, optional
        If True, the input index set I is assumed to be in 1-based index format.
        C will use 1-based indexing then.
        Default is True.

    Returns
    -------
    adm : bool
        True if the input index set is admissible, False otherwise.
    C : ndarray
        The admissible index set in lexicographic order. If the input set is not
        admissible, the missing indices are added to make it admissible.
        If matlab_idx is True, C will use 1-based indexing.

    Notes
    -----
    The function checks each index in the set for admissibility and adds any missing
    indices to ensure the set is admissible. A warning is logged if the set is not
    admissible and indices are added.
    """
    # [adm,C] = CHECK_SET_ADMISSIBILITY(I) checks whether I is an admissible set.
    #       If that is the case, adm=true and C contains I ordered in lexicographic order.
    #       If not, adm=false and C contains I plus the multiindeces needed, again in lexicographic order
    assert is_number_or_list_of_numbers_nonnegative(I)
    if matlab_idx == True:
        # from matlab index numbers to python (I represents an index set here)
        I = matlab_to_python_index(I)
    isarray = hasattr(I, '__iter__')
    I = np.atleast_2d(I)
    C = I.copy()
    # now check amdissibility condition and add what's missing. Print a warning if needed
    row = 0
    row_max = C.shape[0]
    adm = 1  ## True
    while row < row_max:
        idx = C[row,:] #.copy() # [:] returns a view, but copy can be omitted here
        is_adm, _, missing_set = check_index_admissibility(idx, C)
        # if it is not admissible, add what's needed to the bottom of C
        # and increase row_max, so that the new indices will also be checked.
        # So we don't sort the C after the update, otherwise I am not sure I am
        # checking everything
        if is_adm == 0:
            # update C and counter
            C = misc.safe_vstack(C, missing_set)
            row_max = C.shape[0]
            # print a warning
            logger.warning(f"The set is not admissible. Adding: {python_to_matlab_index(missing_set) if matlab_idx else missing_set}")
            # record non admissibility has been found
            adm = 0  ## False
        row = row + 1

    if isarray:
        # finally, sort C
        C,_ = matlab.sortrows(C)
    else:
        C = C.ravel()[0]
    if matlab_idx == True:
        C = python_to_matlab_index(C)
    return adm, C
