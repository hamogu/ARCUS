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
from arcus.instrument.arcus import defaultconf
from ..arfrmf import filename_from_meta, onccd, mkarf, mirr_grat


def test_filename():
    n = filename_from_meta('arf', ARCCHAN='1111', ORDER=3)
    assert n == 'chan_all_+3.arf'

    n = filename_from_meta(ARCCHAN='1111', ORDER=3,
                           TRUEORD=-3, CCDORDER=5)
    assert n == 'chan_all_ccdord_+5_true_-3.fits'

    # After reading a fits file, keywords will be string.
    # Some may have been modified, so test mixture of int and string
    n = filename_from_meta(ARCCHAN='1111', ORDER='+3',
                           TRUEORD='-4', CCDORDER=-5)
    assert n == 'chan_all_ccdord_-5_true_-4.fits'

    # But and order is not confused by itself
    n = filename_from_meta(ARCCHAN='1111', ORDER=-3,
                           TRUEORD=-3, CCDORDER=-3)
    assert n == 'chan_all_ccdord_-3_true_-3.fits'


def test_onccd():
    '''Don't want to hardcode exact locations of CCDs in this test,
    because we might move around CCDs or aimpoints. So, instead, this test
    just checks a few generic properties and wavelengths that are wildly off
    chip.
    '''
    out = onccd(np.arange(200) * u.Angstrom, 1, '1')
    assert out.sum() > 0  # Some wavelength fall on chips
    assert out.sum() < len(out)  # but not all of them

    # chip gaps are different for different orders
    out1 = onccd(np.arange(25, 30, .001) * u.Angstrom, -5, '1')
    out2 = onccd(np.arange(25, 30, .001) * u.Angstrom, -6, '1')
    assert not np.all(out1 == out2)

    # chip gaps are at same position in m * lambda space
    out1 = onccd(np.arange(25, 30, .001) * u.Angstrom, -5, '1')
    out2 = onccd(np.arange(25, 30, .001) * u.Angstrom / 6 * 5, -6, '1')
    assert np.all(out1 == out2)

    # chip gaps are different for different opt ax
    out1 = onccd(np.arange(30, 35, .001) * u.Angstrom, -4, '1')
    out2 = onccd(np.arange(30, 35, .001) * u.Angstrom, -4, '2')
    assert not np.all(out1 == out2)


def test_arf_nonzero():
    '''Arcus design will evolve, so I don't want to test exact numbers here
    but if we are off my orders of magnitude that that's likely a code error.
    '''
    arf = mkarf([23, 24, 25, 26] * u.Angstrom, -5)
    assert np.all(arf['SPECRESP'] > 10 * u.cm**2)
    assert np.all(arf['SPECRESP'] < 400 * u.cm**2)


def test_arf_channels_addup():
    arfall = mkarf([33, 34, 35, 36] * u.Angstrom, -3)

    arflist = []
    for k in defaultconf['pos_opt_ax'].keys():
        arflist.append(mkarf([33, 34, 35, 36] * u.Angstrom, -3, channels=[k]))

    assert np.allclose(arfall['SPECRESP'],
                       sum([a['SPECRESP'] for a in arflist])
                       )


def test_mirrgrat():
    '''Test that mirr_grat works. Don't want to hardcode specific numbers
    (since they will change), but test a range here.
    '''
    aeff = mirr_grat([10, 20, 30] * u.Angstrom, -5)
    assert np.all(aeff < [10, 50, 100] * u.cm**2)
    assert np.all(aeff > [.1, 5., 10.] * u.cm**2)
