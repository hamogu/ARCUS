import datetime
import numpy as np
from astropy.coordinates import SkyCoord

from marxs.source import PointSource, FixedPointing, JitterPointing
import arcus

n_photons = 1e6
outdir = '/melkor/d1/guenther/processing/ARCUS/'

# define position and spectrum of source
mysource = PointSource(coords=SkyCoord(0., 0., unit='deg'),
                       energy={'energy': np.array([0.25, 1.7]),
                               'flux': np.ones(2)},
                       flux=1.)
jitterpointing = JitterPointing(coords=SkyCoord(0., 0., unit='deg'),
                                jitter=arcus.jitter_sigma)
fixedpointing = FixedPointing(coords=SkyCoord(0., 0., unit='deg'))


for i in range(10):
    print 'jitter: {:03d}'.format(i), ' : ', datetime.datetime.now()
    # Ignore geometric area and set number of photons by hand.
    photons = mysource.generate_photons(n_photons)
    photons = jitterpointing(photons)
    photons = arcus.arcus4(photons)

    photons.write(outdir + 'flatspecjitter{:03d}.fits'.format(i), overwrite=True)

for i in range(10):
    print 'fixed: {:03d}'.format(i), ' : ', datetime.datetime.now()
    # Ignore geometric area and set number of photons by hand.
    photons = mysource.generate_photons(n_photons)
    photons = fixedpointing(photons)
    photons = arcus.arcus4(photons)

    photons.write(outdir + 'flatspecfixed{:03d}.fits'.format(i), overwrite=True)
