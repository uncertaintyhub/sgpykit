import numpy as np

from sgpykit.tools.polynomials_functions.lagr_eval_fast import lagr_eval_fast
from sgpykit.util.struct import Struct

def interpolate_on_sparse_grid(S, Sr, function_on_grid, non_grid_points):
    """
    Interpolate a function on a sparse grid.

    Evaluates the sparse grid polynomial approximation (surrogate model) on a
    generic point of the parameters space.

    Parameters
    ----------
    S : list of Struct or Struct
        Sparse grid structure(s). If a single Struct is provided, it is converted
        to a list.
    Sr : Struct
        Reduced version of the sparse grid S.
    function_on_grid : ndarray
        Matrix containing the evaluation of the function on the points of Sr.
        Dimensions: V x number_of_points_in_the_sparse_grid.
    non_grid_points : ndarray
        Set of points where the sparse grid polynomial approximation is to be
        evaluated. Each column is a different point.

    Returns
    -------
    ndarray
        Matrix containing the evaluation of the vector-valued function in each of
        the non_grid_points. Dimensions: V x number_of_non_grid_point.
    """
    function_on_grid = np.atleast_2d(function_on_grid)
    N, nb_points_Sr = Sr.knots.shape
    if function_on_grid.shape[1] != nb_points_Sr:
        # if this condition is true, then function_on_grid stores
        # evaluation as rows rather than as columns and we stop the
        # function
        raise ValueError('Incompatible sizes.')

    # similarly, we have changed the dimensions of non_grid_points to follow
    # the convention that points are stored as columns. Again, we throw an
    # error if the other convention is detected (this also prevents accidental input errors)
    if non_grid_points.shape[0] != N:
        raise ValueError('Incompatible sizes.')

    # the other sizes.
    V = function_on_grid.shape[0]
    nb_pts = non_grid_points.shape[1]
    f_values = np.zeros((nb_pts, V))
    function_on_grid = np.transpose(function_on_grid)
    if isinstance(S, Struct):
        S = [S]
    nb_grids = len(S)
    # I need to go back from each point of each grid to its corresponding value in function_on_grid (F comes from an evaluation over a reduced grid)
    # I only have a global mapping from [S.knots] to Sr.knots, so I need a global counter that scrolls [S.knots]
    global_knot_counter = 0
    # loop over the grids
    for i in range(nb_grids):
        # some of the grids in S structure are empty, so I skip it
        if len(S[i].weights) == 0:
            continue

        # this is the set of points where I build the tensor lagrange function
        knots = S[i].knots
        # I will need the knots in each dimension separately, to collocate the lagrange function.
        # I compute them once for all. As the number of knots is different in each direction, I use a cell array
        # we had here
        # dimtot=size(knots,1);
        # clearly dimtot==N, hence we sobstitute it everywhere
        # the following lines are also no longer needed
        #  knots_per_dim=np.zeros(N);
        #  for dim=1:N
        #      knots_per_dim{dim}=unique(knots[dim,:]);
        #  end
        knots_per_dim = S[i].knots_per_dim
        # I also have to take into account the coefficient of the sparse grid
        coeff = S[i].coeff
        # we could just loop on the interpolation points of the current tensor grid,
        # and for each one evaluate the corresponding lagrangian polynomial on
        # the set of non_grid_points, but this is not convenient. Actually doing this we recompute the same thing over and over!
        # think e.g. of the lagrangian functions for the knot [x1 y1 z1] and
        # for the knot [x1 y1 z2]: the parts on x and y are the same!
        # Therefore we evaluate each monodim_lagr just once and save all these
        # evaluations in a cell array. We will combine them suitably afterward.
        # Such cell array stores one matrix per direction, i.e. N matrices.
        # In turn, the n-th matrix contains the evaluations of each monodimensional lagrange polynomial
        # in direction n on all the n-th coordinates of the non_grid_points.
        # Its dimension will be therefore (number_of_points_to_evaluate) X (number_of_lagrangian_polynomials_in_the_nth_direction)
        # This is actually all the information that we will need to combine to
        # get the final interpolant value.
        mono_lagr_eval = []
        # loop on directions
        for dim in range(N):
            # this is how many grid points in the current direction for the
            # current tensor grid
            # K=length(knots_per_dim{dim});
            K = S[i].m[dim]
            # allocate space for evaluations. Since I will be accessing it one lagrangian polynomial at a time
            # i.e. one knot at a time, it's better to have all information for the same lagrange polynomial
            # on the same column, for speed purposes. Moreover, note that whenever K=1 (one point only in a direction)
            # then the lagrange polynomial is identically one
            mono_lagr_eval.append( np.ones((nb_pts, K)) )
            if K > 1:
                # loop on each node of the current dimension and evaluate the corresponding monodim lagr polynomial.
                # We will need an auxiliary vector to pick up the current knot (where the lagr pol is centered) and
                # the remaining knots (where the lagr pol is zero). Here we see that mono_lagr_eval it's written
                # one column at a time
                aux = np.arange(K)
                for k in aux:
                    current_knot = knots_per_dim[dim][k]
                    other_knots = np.array(knots_per_dim[dim])[aux != k]
                    mono_lagr_eval[dim][:, k] = lagr_eval_fast(current_knot, other_knots,
                                                               K - 1, non_grid_points[dim, :], (nb_pts, 1))

        # now put everything together. We have to take the tensor product of
        # each of the monodim lagr pol we have evaluated. That is, we have to
        # pick one column for each matrix in the cell array and dot-multiply them.
        # all the possible combinations have to be generated !
        # once this is done, we have the evaluation of each multidim lagr
        # polynomial on the non_grid_points, which we will then multiply by the
        # corresponding nodal value and eventually sum everything up.
        # We start by generating the combination of column we need to take. We actually don't
        # need to generate them, but only to recover it from the matrix knots,
        # which already contains all the points of the grid, i.e. all the
        # combinations of 1D points!
        # Given a matrix of points like
        # knots=[a1 a1 b1 b1 a1 a1 b1 b1 ...
        #        a2 b2 a2 b2 .....
        # combi is
        # combi=[1 1 2 2 1 1 2 2 ...
        #        1 2 1 2 ......
        # again, we exploit the fact that the minimum entry of combi is 1 and that for many directions
        # there is only one point, so if we init combi with ones we're good in many cases
        combi = np.zeros((N, S[i].size), dtype='int') ## used for indices, so start with 0
        # the easiest way to recover combi from knots is to proceed one dimension at a time,
        # and mark with a different label (1,2,...K) all the equal points. We need of course as many labels
        # as the number of different points in each dir!
        for dim in range(N):
            # this is how many points per direction
            # K=length(knots_per_dim{dim});
            K = S[i].m[dim]
            # we start from a row of zeroes and we place 1....K in the right
            # positions by summations (each element of the row will be written
            # only once!). Since 1 are already in place, we proceed to place 2 and higher, but only if needed
            if K > 1:
                for k in range(1,K):
                    # here we add to the row of "1" either 0 or (k-1) so we get k where needed
                    combi[dim, :] += k * (knots[dim, :] == knots_per_dim[dim][k])
        # Now we can do the dot-multiplications among rows, the
        # multiplication by nodal values and the final sum! We proceed one
        # knot at a time
        for kk in range(S[i].size):
            # dot-multiply all the lagrangian functions according to the
            # combi represented by the current knot. The result F_LOC is a
            # column vector
            tmp = np.array(mono_lagr_eval[0])
            tmpidx = combi[0, kk]
            f_loc = tmp[:, tmpidx]
            for dim in range(1, N):
                f_loc = f_loc * mono_lagr_eval[dim][:, combi[dim, kk]]
            # recover F, the corresponding value for the interpolating function in function_on_grid, with the global counter
            position = Sr.n[global_knot_counter]
            F_value = function_on_grid[position, :]
            # add the contribution of this knot to the sparse interpolation.
            # Its contribution is a matrix, since I am evaluating it on a bunch
            # of non_grid_points (one per row of f_values, will be trasposed later),
            # and my function to be evaluated is vector-valued
            # which gives a bunch of rows in columns in f_values.
            # we generate this matrix as outer product of f_loc (column vector with lagr pol values) and F_value
            # (row vector with function evaluations)
            f_loc = np.atleast_2d(f_loc).T  # TODO: can we further simplify this part?
            F_value = np.squeeze(F_value)
            f_values = f_values + coeff * f_loc * F_value
            # update global counter
            global_knot_counter = global_knot_counter + 1

    # finally, transpose to comply with output orientation
    f_values = np.transpose(f_values)
    return f_values
