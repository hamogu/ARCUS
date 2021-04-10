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
from astropy.coordinates import SkyCoord
import astropy.units as u
from ..instrument import Arcus
from ..defaults import DefaultSource, DefaultPointing


def test_default_source():
    '''Check that the orders look reasonable.

    This test tries to be generic so that coordinate system etc. can be
    changed later, but still check that all light is focused to
    one point to detect error in setting up.
    '''
    instrument = Arcus(channels=['1'])
    photons = DefaultSource().generate_photons(1e4 * u.s)
    photons = DefaultPointing()(photons)
    photons = instrument(photons)

    ind = (photons['order'] == 5) & np.isfinite(photons['det_x']) & (photons['probability'] > 0)
    if ind.sum() > 100:
        assert np.std(photons['det_y'][ind]) < 1
        assert np.std(photons['det_x'][ind]) < 1
        assert np.std(photons['det_x'][ind]) < np.std(photons['det_y'][ind])


def test_default_source_parameters():
    '''Make sure that parameters are passed through even if they are different
    from defaults.'''
    s = DefaultSource(energy=0.5 * u.keV)
    p = s(10 * u.s)
    assert np.allclose(p['energy'], 0.5)

    s = DefaultSource(coords=SkyCoord(30. * u.deg, 0. * u.deg))
    p = s(10 * u.s)
    p = DefaultPointing()(p)
    assert np.allclose(np.abs(p['dir'][:, 2]), np.cos(np.deg2rad(30)))
