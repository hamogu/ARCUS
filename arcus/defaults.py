'''Provide simple defaults for a Arcus simulation

This module sets up a few simple defaults for ray-tracing things with Arcus.
It provides a source and pointing class for an on-axis pointing with some
sensibe defaults for energy, flux, jitter etc.
The purpose if this module is to save some typing when running simulations
from the command line and to make is easier to keep many different test
scripts in sync to all use the same parameters.
'''
from marxs.source import PointSource, JitterPointing
import astropy.units as u
from astropy.coordinates import SkyCoord
from .instrument.arcus import xyz2zxy, jitter_sigma

coords = SkyCoord(30. * u.deg, 30. * u.deg)

sourcekwargs = {'coords': coords,
                'energy': 0.5 * u.keV,
                'flux': 1. / u.s / u.cm**2
                }
pointingkwargs = {'coords': coords,
                  'reference_transform': xyz2zxy,
                  'jitter': jitter_sigma
                  }


class DefaultSource(PointSource):
    '''Default astronomical source for Arcus.

    This source will work with no parameters passed in and generate photons
    with a sensible pre-set for Arucs simulations (e.g. a fixed energy in
    the range that is interesting for Arcus).

    Parameters
    ----------
    Set any parameters that the base class would accept. Parameters are
    passed right through, the only difference in behaviour to the base class is
    that defaults are provided for all required parameters (e.g. the
    coordiantes on the sky).
    '''
    def __init__(self, **kwargs):
        for k in sourcekwargs:
            if k not in kwargs:
                kwargs[k] = sourcekwargs[k]
        super(DefaultSource, self).__init__(**kwargs)


class DefaultPointing(JitterPointing):
    '''Default astronomical source for Arcus.

    This pointing will work with no parameters passed in and it will set
    sensible defaults for an Arcus simulation (e.g. photons passing along
    the z-axis of the coordiante system for an on-axis source).

    Parameters
    ----------
    Set any parameters that the base class would accept. Parameters are
    passed right through, the only difference in behaviour to the base class is
    that defaults are provided for all required parameters (e.g. the jitter and
    the pointing direction.
    '''

    def __init__(self, **kwargs):
        for k in pointingkwargs:
            if k not in kwargs:
                kwargs[k] = pointingkwargs[k]
        super(DefaultPointing, self).__init__(**kwargs)
