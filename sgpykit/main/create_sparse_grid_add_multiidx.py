import logging
import numpy as np

from sgpykit.src.delta_to_combitec import delta_to_combitec
from sgpykit.src.find_lexicographic import find_lexicographic
from sgpykit.src.tensor_grid import tensor_grid
from sgpykit.util import matlab
from sgpykit.util.misc import to_array, apply_lev2knots

logger = logging.getLogger(__name__)


def create_sparse_grid_add_multiidx(new_idx, S_in, I_in, coeff_in, knots, lev2knots):
    """
    Produce a grid obtained by adding a single multi-idx to a previously existing grid.

    Parameters
    ----------
    new_idx : array_like
        An index that must be admissible with respect to the index set I_in.
        This condition won't be checked by the function.
    S_in : array_like
        A sparse grid to which new_idx should be added.
    I_in : array_like
        The index set that was used to create S_in, either implicitly (by defining
        the rule in CREATE_SPARSE_GRID) or explicitly (by using
        CREATE_SPARSE_GRID_MULTIIDX). Note that this cannot be produced as the
        union of the indices S_in.idx, because S_in contains only tensors whose
        coefficient in the combination technique is non-zero, while here we need
        them all.
    coeff_in : array_like
        The vector of coefficients of the combination technique obtained on I_in.
        Note that this vector of coefficients *MUST* include also zeros if a row
        of I_in has coeff zero in the combination technique. Therefore, coeff_in
        cannot be taken as [S_in.coeff]. Instead, use coeff_in =
        COMBINATION_TECHNIQUE(I_in).
    knots : array_like or callable
        The usual argument to specify knots.
    lev2knots : array_like or callable
        The usual argument to specify lev2knots.

    Returns
    -------
    S : array_like
        The new sparse grid.
    I : array_like
        The new multiidx set, i.e., I = sortrows([I_in; new_idx]).
    coeff : array_like
        The updated vector of coefficients of the combination technique.
    """
    # sort rows of the index set after adding the new index
    combined = np.vstack([I_in, new_idx])
    I,sorter = matlab.sortrows(combined)
    # add 0 to the new coeff_G in its right place (+1 will come later on)
    coeff = np.append(coeff_in, 0)[sorter]

    # first a check on C_old being sorted. Observe that the function sortrows used is very efficient
    # so the cost of this preliminary analysis is negligible (e.g. it takes 0.02 sec to verify
    # that a TD set with w=5 and N=30, i.e. ~340600 indices is sorted,  and only 0.00027 sec that
    # the matrix A=randi(20,300000,30) is unsorted.

    if not matlab.issorted(I_in,'rows') :
        raise ValueError('the multiindex set C_old is not sorted')

    N = I_in.shape[1]
    # if knots and  lev2knots are simple function, we replicate them in a cell
    if callable(knots):
        knots = to_array(knots, N)

    if callable(lev2knots):
        lev2knots = to_array(lev2knots, N)

    #-----------------------------------------------
    # generate contributions of jj to combitec. They are already lexicosorted, jj is the last row
    Tensors_jj = delta_to_combitec(new_idx)
    # look them up in G and for each of them add +-1. Because both Tensor and GG are sorted,
    # restrict the search at each iteration

    nb_tens = Tensors_jj.shape[0]
    from_where = 0
    pos = 0
    for t in range(nb_tens):
        tt = Tensors_jj[t, :]
        found, rel_pos, _ = find_lexicographic(tt, I[from_where:, :], 'nocheck')
        if found:
            pos = from_where + rel_pos
        if not found:
            raise RuntimeError('I was supposed to find this tensor!')
        coeff_tt = (-1) ** np.sum(new_idx - tt)
        coeff[pos] = coeff[pos] + coeff_tt
        from_where = pos + 1

    # now we can store only those grids who survived, i.e. coeff~=0
    #------------------------------------------------------

    nb_grids = np.sum(coeff != 0)
    empty_cells = matlab.cell((1,nb_grids))
    S = matlab.struct('knots',empty_cells,'weights',empty_cells,'size',empty_cells,'knots_per_dim',empty_cells,'m',empty_cells)
    fieldnms = matlab.fieldnames(S)

    coeff_condensed = np.zeros(nb_grids)
    ss = 0
    # for each nonzero coeff, generate the tensor grid and store it. If possible, recycle from S_old. Note that C_old
    # stores all idx of S_old, even those with 0 coeff. So I need to extract the information on the tensors that I have
    # in S_old as follows

    logger.debug('build sparse grid with tensor grid recycling')

    nb_S_old_grids = len(S_in)
    C_old_nnz = np.hstack(S_in.idx).reshape(N, nb_S_old_grids, order='F').T  # TODO review

    for j in range(len(coeff)):
        if coeff[j] != 0:
            i = I[j,:]
            found,pos,_ = find_lexicographic(i,C_old_nnz,'nocheck')
            if found:
                # Note that at this point elements of S are tensor grids while S_old is a sparse grid therefore it has additional fields
                # (coeff, idx). We thus need to copy field by field otherwise we'll have "assignment between dissimilar
                # structures" error. We use dynamic filed names to this end
                for fn in fieldnms:
                    setattr(S[ss],fn,getattr(S_in[pos],fn))
                # however we need to fix the weights. Indeed, they are stored in S_old as weights*coeff, so we need to reverse
                # that multiplication
                S[ss].weights = S[ss].weights / S_in[pos].coeff
            else:
                m = apply_lev2knots(i,lev2knots,N)
                S[ss] = tensor_grid(N,m,knots)
            S[ss].weights = S[ss].weights * coeff[j]
            coeff_condensed[ss] = coeff[j]
            ss = ss + 1

    # now store the coeff value. It has to be stored after the first loop, because tensor_grid returns a grid
    # WITHOUT coeff field, and Matlab would throw an error (Subscripted assignment between dissimilar structures)

    for ss in range(nb_grids):
        S[ss].coeff = coeff_condensed[ss]

    # similarly for the multiidx generating each tensor grid
    ss = 0
    for j in range(len(coeff)):
        if coeff[j] != 0:
            i = I[j,:]
            S[ss].idx = i
            ss = ss + 1

    return S,I,coeff
