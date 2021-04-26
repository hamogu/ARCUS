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
from ..osip import FixedWidthOSIP, FixedFractionOSIP, FractionalDistanceOSIP

import pytest


osipf = FixedWidthOSIP(40 * u.eV)
osipp = FixedFractionOSIP(0.7)
osipd = FractionalDistanceOSIP()


def test_osip_factor_width_unit():
    '''check units work for osip_width'''
    osip1 = FixedWidthOSIP(40 * u.eV)
    osip2 = FixedWidthOSIP(0.04 * u.keV)
    assert np.all(osip1.osip_factor([10] * u.Angstrom, -5, -5) ==
                  osip2.osip_factor([10] * u.Angstrom, -5, -5))


@pytest.mark.parametrize("thisosip", [osipf, osipp, osipd])
def test_osip_factor_wave_unit(thisosip):
    '''Check input works for wave or energy'''
    wave = [20, 21, 22, 23] * u.Angstrom
    energ = wave.to(u.keV, equivalencies=u.spectral())
    assert np.all(thisosip.osip_factor(wave, -5, -5) ==
                  thisosip.osip_factor(energ, -5, -5))


def test_osip_factor():
    '''test extreme values that do not depend on CCD resolution'''
    assert FixedWidthOSIP(0 * u.eV).osip_factor([10] * u.Angstrom, -5, -5) == 0
    wide = FixedWidthOSIP(10 * u.keV)
    assert np.allclose(wide.osip_factor([10] * u.Angstrom, -5, -5), 1)
    '''test with fixed sigma'''
    def sig(args):
        return 40 * u.eV
    myosip = FixedWidthOSIP(40 * u.eV, sig_ccd=sig)

    assert np.allclose(myosip.osip_factor([10] * u.Angstrom, -5, -5),
                       0.6827, rtol=1e-4)


@pytest.mark.parametrize("thisosip", [osipf, osipp, osipd])
def test_osip_factor_orders_on_different_sides(thisosip):
    '''If one order is positive and the other negative, then
    the signal is diffracted to opposite sides, so there is no
    contamination.
    '''
    assert thisosip.osip_factor([10] * u.Angstrom, -5, 5) == 0
    assert thisosip.osip_factor([10] * u.Angstrom, 1, -1) == 0
    assert thisosip.osip_factor([10] * u.Angstrom, 1, 0) == 0


@pytest.mark.parametrize("thisosip", [osipf, osipp, osipd])
def test_symmetry(thisosip):
    '''offset_orders +1 and -1 should have the same OSIP factors
    if OSIP is symmetric'''
    up = thisosip.osip_factor([10, 20, 30] * u.Angstrom, -5, -6)
    down = thisosip.osip_factor([10, 20, 30] * u.Angstrom, -5, -4)
    assert np.allclose(up, down, atol=1e-4)
