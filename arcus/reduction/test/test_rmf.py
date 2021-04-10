#    Copyright (C) 2021  Massachusetts Institute of Technology
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from ..ogip import RMF


def test_compress_rmf_matrix():
    arr = np.array([0., 1e-9, .1, .2, .1, 1e-3, .1])
    ngrp, fchan, nchan, mat = RMF.arr_to_rmf_matrix_row(arr, 1,
                                                        threshold=1e-2)
    assert ngrp == 2
    assert fchan == [3, 7]
    assert nchan == [3, 1]
    assert mat == [.1, .2, .1, .1]

    ngrp, fchan, nchan, mat = RMF.arr_to_rmf_matrix_row(arr, 0,
                                                        threshold=1e-2)
    assert ngrp == 2
    assert fchan == [2, 6]
    assert nchan == [3, 1]
    assert mat == [.1, .2, .1, .1]
