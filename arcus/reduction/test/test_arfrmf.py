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
import astropy.units as u
from ..arfrmf import filename_from_meta, onccd


def test_filename():
    n = filename_from_meta('root', 'arf', CHANNEL='1111', ORDER=3)
    assert n == 'root_chan_all_+3.arf'

    n = filename_from_meta(CHANNEL='1111', ORDER=3,
                           RFLORDER=-4, CCDORDER=-5)
    assert n == 'root_chan_all_-4_confusedby_-5.fits'

    # After reading a fits file, keywords will be string.
    # Some may have been modified, so test mixture of int and string
    n = filename_from_meta(CHANNEL='1111', ORDER='+3',
                           RFLORDER='-4', CCDORDER=-5)
    assert n == 'root_chan_all_-4_confusedby_+5.fits'


def test_onccd():
    '''Don't want to hardcode exact locations of CCDs in this test,
    because we might move around CCDs or aimpoints. So, instead, this test
    just checks a few generic properties and wavelengths that are wildly off
    chip.
    '''
    out = onccd(np.arange(200) * u.Angstrom, 1, '1')
    assert out.sum() > 0  # Some wavelength fall on chips
    assert out.sum() < len(out)  # but not all of them

    out1 = onccd(np.arange(25, 30, .001) * u.Angstrom, -5, '1')
    out2 = onccd(np.arange(25, 30, .001) * u.Angstrom, -6, '1')
    assert out1 != out2  # chip gaps are different for different orders

    out1 = onccd(np.arange(25, 30, .001) * u.Angstrom, -5, '1')
    out2 = onccd(np.arange(25, 30, .001) * u.Angstrom / 6 * 5, -6, '1')
    assert out1 == out2  # chip gaps are at same position in m * lambda space

    out1 = onccd(np.arange(30, 35, .001) * u.Angstrom, -4, '1')
    out2 = onccd(np.arange(30, 35, .001) * u.Angstrom, -4, '2')
    assert out1 != out2  # chip gaps are different for different opt ax
