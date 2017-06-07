import time
import os
import tempfile
import numpy as np

from marxs.source import PointSource, FixedPointing
from marxs import utils
from astropy.table import Table
import astropy.units as u
from astropy import table
from astropy.utils.metadata import enable_merge_strategies
from astropy.io import fits
import arcus

n_photons_list = [1e4, 1e5, 1e6, 1e7]

wave = np.arange(8., 50., 0.5) * u.Angstrom
energies = wave.to(u.keV, equivalencies=u.spectral()).value

for n_photons in n_photons_list:
    t0 = time.time()
    mysource = PointSource((0., 0.), energy=0.5, flux=1.)
    photons = mysource.generate_photons(n_photons / 2)

    mypointing = FixedPointing(coords = (0., 0.))
    photons = mypointing(photons)
    photons = arcus.arcus_joern(photons)

    photonsm = mysource.generate_photons(n_photons / 2)
    photonsm = mypointing(photonsm)
    photonsm = arcus.arcus_joernm(photonsm)
    photonsm['aperture'] += 2

    with enable_merge_strategies(utils.MergeIdentical):
        out = table.vstack([photons, photonsm])

        photons.write(tempfile.NamedTemporaryFile(), format='fits')
    runtime = time.time() - t0
    print 'n: {0:5.1e}, total time: {1:5.2f} s, time per photon: {2:5.1e} s'.format(n_photons, runtime, runtime / n_photons)
