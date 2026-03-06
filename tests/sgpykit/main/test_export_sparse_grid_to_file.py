import pytest
import numpy as np
import os

import sgpykit as sg


def test_export_sparse_grid_to_file():
    N = 3

    aa = [4, 1, -2]
    bb = [6, 5, -1]

    knots1 = lambda n: sg.knots_CC(n, aa[0], bb[0], 'nonprob')
    knots2 = lambda n: sg.knots_CC(n, aa[1], bb[1], 'nonprob')
    knots3 = lambda n: sg.knots_uniform(n, aa[2], bb[2], 'nonprob')

    w = 2
    S, _ = sg.create_sparse_grid(N, w, [knots1, knots2, knots3], sg.lev2knots_doubling)
    Sr = sg.reduce_sparse_grid(S)

    # Save points to 'points.dat'. The first row actually contains two integer
    # values, i.e., Sr.size and N
    sg.export_sparse_grid_to_file(Sr)

    # Save points to 'mygrid.dat'
    sg.export_sparse_grid_to_file(Sr, 'mygrid2.dat')

    # Save points and weights to 'mygrid_with_weights.dat'
    sg.export_sparse_grid_to_file(Sr, 'mygrid_with_weights2.dat', with_weights=True)

    def test_file_and_remove_on_success(fname):
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, fname)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    success = first_line == "27 3"
            except IOError:
                return False
            if success:
                os.remove(file_path)
                return True
        return False

    assert test_file_and_remove_on_success('mygrid2.dat')
    assert test_file_and_remove_on_success('mygrid_with_weights2.dat')
