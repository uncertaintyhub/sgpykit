# Also check test_matlab.py
#
from sgpykit.util.matlab import cell


def test_cellInit():
    nb = 4
    cells = cell(nb)
    assert cells.shape[0] == nb and cells.shape[1] == nb
    cells_2 = cell((1, nb))
    assert cells_2.shape[0] == 1 and cells_2.shape[1] == nb
    cells_3 = cell((nb, 1))
    assert cells_3.shape[0] == nb and cells_3.shape[1] == 1


def test_cellIndex():
    nb = 4
    cells = cell((1, nb))
    cells[2] = -1
    assert cells[0,2] == -1
    cells_2 = cell((nb, 2))
    cells_2[2,1] = 4
    assert cells_2[5] == 4
