# Differences between Matlab and NumPy or SGMK and sgpykit

| MATLAB / SGMK                                 | NumPy / sgpykit                                                                                                                                                                                                                                                                                                                                                 |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| The basic type, even for scalars, is a multidimensional array. Array assignments are stored as 2D arrays of double precision floating point numbers per default. | Scalars are not arrays. The default array type is float64, but you can specify the dtype and number of dimensions.                                                                                                                                                                                                                                              |
| Numbers indices from 1.                                                                                                        | Numbers indices from 0.                                                                                                                                                                                                                                                                                                                                         |
| Array slicing uses pass-by-value semantics, with a lazy copy-on-write scheme to prevent creating copies until needed. Slicing operations copy parts of the array. | Array slicing uses pass-by-reference, so slicing operations are views into an array and do not copy the data.                                                                                                                                                                                                                                                   |
| Always allows multi-dimensional arrays to be accessed using scalar or linear indices (linear indexing).                         | Does not support linear indexing for multi-dimensional arrays directly. To mimic linear indexing, you must reshape to a 1D array, operate, and reshape back. There is also `some_array.flat[i]` to access the array in a 1D view, but it is row-major ordered. `np.ravel(some_array, order='F')` flattens the array by a given order and tries to avoid a copy. |
| `reshape` uses Fortran order (column-major) by default.                                                                        | `reshape` uses C order (row-major) by default, but can use Fortran order with `order='F'`.                                                                                                                                                                                                                                                                      |
| `reshape` returns a copy of the data.                                                                                          | `reshape` usually returns a view onto the same storage (no data copy), unless not possible. Use `.copy()` to force a copy.                                                                                                                                                                                                                                      |
| `find()` returns linear indices.                                                                                               | `numpy.where` returns a tuple of arrays (one per dimension), not linear indices. There is a `matlab.find()` in sgpykit, but such re-implementations lack functionality compared to matlab.                                                                                                                                                                      |
| `[some_struct_array.some_array]` | `x = reshape_nested_lists_to_nrows(x, nrows=2)` (from `sgpykit.util.misc`)                                                                                                                                                                                                                                                                                      |
| Ignoring return objects | Placeholder `_` is required, eg. `x, _ = matlab.sort(x)` or `fs, *_ = evaluate_on_sparse_grid(f,S)` for unknown number of placeholders.                                                                                                                                                                                                                         |
| `a:b` includes both endpoints. | `numpy.arange(a, b)` excludes the endpoint b, while `numpy.linspace(a, b, num)` includes both endpoints. Python `range(a, b)` does not include b. So `my_list[a:b]` does not include the element at index b                                                                                                                                                     |
| structs, cell operator `{}` | sgpykit has only limited support using `Struct`, `StructArray`, `Cell` and `struct()` and `cell()`. The function `ce()` mimics matlab's `{}` operator to create new cells.                                                                                                                                                                                      |
| SGMK often looks for number of input arguments | sgpykit/python allows keyword args in no specific order, so use positional/named arguments.                                                                                                                                                                                                                                                                     |

For more information: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

## API Differences (Incomplete)

| Feature/Aspect               | MATLAB                                 | NumPy                                                                             |
|------------------------------|----------------------------------------|-----------------------------------------------------------------------------------|
| **Array Indexing**           | 1-based indexing                       | 0-based indexing                                                                  |
| **Array Creation**           | `[1, 2, 3]`                            | `np.array([1, 2, 3])` (1D) or `np.array([[1,2,3]])` (2D)                          |
| **Array Range**              | `1:10`                                 | `np.arange(1, 11)`                                                                |
| **Array Shape**              | `size(A)`                              | `A.shape`                                                                         |
| **Array Size**               | `numel(A)`                             | `A.size`                                                                          |
| **Array Reshape**            | `reshape(A, m, n)`                     | `A.reshape(m, n, order="F")` or `np.reshape(A, (m,n), order="F")`                 |
| **Array Shrink**             |  `some_array(idx:end)=[];` | `some_array.resize(idx)`                                                          |
| **Array Removal**            | `some_array(idx)=[];` | `some_array = np.delete(some_array, dx)`                                          |
| **Array Transpose**          | `A'`                                   | `np.atleast_2d(A).T`                                                              |
| **Array Concatenation**      | `[A, B]`                               | `np.concatenate((A, B), axis=1)` or `np.hstack((A, B))`                           |
| **Array Concatenation**      | `[some_matrix; vector]`                | `np.vstack((some_matrix, vector))`                                                |
| **Array Slicing**            | `A(1:3, 2:4)`                          | `A[0:3, 1:4]`                                                                     |
| **Element-wise Operations**  | `A .* B`                               | `A * B`                                                                           |
| **Matrix Multiplication**    | `A * B`                                | `A @ B` or `np.dot(A, B)`                                                         |
| **Linear Algebra**           | `inv(A)`                               | `np.linalg.inv(A)`                                                                |
| **Eigenvalues/Eigenvectors** | `W,X=eig(A)`                           | `W,x=util.matlab.eig(A)` (x already a vector)                                     |
| **Solving Linear Systems**   | `A \ b`                                | `np.linalg.solve(A, b)`                                                           |
| **Least Squares**            | `A \ b`                                | `np.linalg.lstsq(A, b)`                                                           |
| **Matrix Determinant**       | `det(A)`                               | `np.linalg.det(A)`                                                                |
| **Matrix Rank**              | `rank(A)`                              | `np.linalg.matrix_rank(A)`                                                        |
| **Matrix Norm**              | `norm(A)`                              | `np.linalg.norm(A)`                                                               |
| **Summation**                | `sum(A)`                               | `np.sum(A, axis=0)`                                                                       |
| **Mean**                     | `mean(A)`                              | `np.mean(A, axis=0)`                                                                      |
| **Standard Deviation**       | `std(A)`                               | `np.std(A, axis=0)`                                                                       |
| **Variance**                 | `var(A)`                               | `np.var(A, axis=0)`                                                                       |
| **Maximum Value**            | `max(A)`                               | `np.max(A, axis=0)`                                                                       |
| **Minimum Value**            | `min(A)`                               | `np.min(A, axis=0)`                                                                       |
| **Sorting**                  | `sort(A)`                              | `util.matlab.sort(A)` (limited)                                                   |
| **Sorting**                  | `sortrows(A)`                          | `util.matlab.sortrows(A)` (limited)                                               |
| **Unique Elements**          | `unique(A)`                            | `util.matlab.unique(A)` (limited)                                                 |
| **Logical Operations**       | `A & B`                                | `np.logical_and(A, B)`                                                            |
| **Element-wise Comparison**  | `A == B`                               | `np.equal(A, B)`, also check `np.allclose()`                                      |
| **Broadcasting**             | Limited broadcasting                   | Full broadcasting support                                                         |
| **Function Handles**         | `@function_name`                       | `function_name`                                                                   |
| **Anonymous Functions**      | `@(x) x^2`                             | `lambda x: x**2`                                                                  |
| **Plotting**                 | `plot(x, y)`                           | `sg.plot(ax, x, y)` (see tutorial notebook)                                       |
| **Subplots**                 | `subplot(m, n, p)`                     | --                                                                                |
| **Figure Creation**          | `figure`                               | `sg.figure_create()`                                                              |
| **Saving Figures**           | `saveas(gcf, 'filename.png')`          | `plt.savefig('filename.png')`                                                     |
| **Structures**               | `struct('field1', value1)`             | `sg.matlab.struct(field1=value1)`                                                 |
| **Cell Arrays**              | `{obj1, obj2}`                     | `sg.matlab.ce(obj1, obj2)` or `[obj1,obj2]` if cell type is not really necessary. |


## Cells and Structure Arrays in sgpykit

- sgpykit uses cells-like objects internally similar to SGMK, but they might be removed in a later release
- sparse grids are structure arrays similar to Matlab, but have less features than in Matlab, 
  - e.g. inhomogeneous structure arrays cannot be reshaped directly like an array in Matlab,
  - for example `S.knots` could be a list of lists with different shapes ...
  - ... such a multi-nested inhomogeneous list still can be reshaped, e.g. by `sg.misc.reshape_nested_lists_to_nrows`
  - a list is not a numpy array, i.e. a typecast like np.asarray is then necessary to apply vector operations