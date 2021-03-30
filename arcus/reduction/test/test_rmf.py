from ..ogip import RMF


def test_compress_rmf_matrix():
    arr = [0., 1e-9, .1, .2, .1, 1e-3, .1]
    ngrp, fchan, nchan, mat = RMF.arr_to_rmf_matrix_row(arr, 1e-2)
    assert ngrp == 2
    assert fchan == [3, 7]
    assert nchan == [3, 1]
    assert mat == [.1, .2, .1, .1]
