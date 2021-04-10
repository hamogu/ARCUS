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
from ..osip import osip_factor, sig_ccd


def test_osip_factor_width_unit():
    '''check units work for osip_width'''
    assert np.all(osip_factor([10] * u.Angstrom, -5, -5, sig_ccd, 40 * u.eV) ==
                  osip_factor([10] * u.Angstrom, -5, -5, sig_ccd, 0.04 * u.keV))


def test_osip_factor_wave_unit():
    '''Check input works for wave or energy'''
    wave = [20] * u.Angstrom
    energ = wave.to(u.keV, equivalencies=u.spectral())
    assert np.all(osip_factor(wave, -5, -5, sig_ccd, 40 * u.eV) ==
                  osip_factor(energ, -5, -5, sig_ccd, 40 * u.eV))


def test_osip_factor():
    '''test extreme values that do not depend on CCD resolution'''
    assert osip_factor([10] * u.Angstrom, -5, -5, sig_ccd, 0 * u.eV) == 0
    assert np.allclose(osip_factor([10] * u.Angstrom, -5, -5, sig_ccd,
                                  10 * u.KeV), 1)
    '''test with fixed sigma'''
    def sig(args):
        return 40 * u.eV

    assert np.allclose(osip_factor([10] * u.Angstrom, -5, -5, sig,
                                   40 * u.eV), 0.6827, rtol=1e-4)
