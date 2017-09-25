import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from .. import Arcus
from ..defaults import DefaultSource, DefaultPointing


def test_default_source():
    '''Check that the orders look reasonable.

    This test tries to be generic so that coordinate system etc. can be
    changed later, but still check that all light is focused to
    one point to detect error in setting up.
    '''
    instrument = Arcus(channels=['1'])
    photons = DefaultSource().generate_photons(1e4)
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
    s = DefaultSource(energy=0.5)
    p = s(10)
    assert np.allclose(p['energy'], 0.5)

    s = DefaultSource(coords=SkyCoord(30. * u.deg, 0. * u.deg))
    p = s(10)
    p = DefaultPointing()(p)
    assert np.allclose(np.abs(p['dir'][:, 2]), np.cos(np.deg2rad(30)))
