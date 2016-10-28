import os
import numpy as np
from astropy import table
import astropy.units as u
from astropy.utils.metadata import enable_merge_strategies

from marxs.source import PointSource, FixedPointing
from marxs import utils

import arcus

n_photons = 1e4
wave = np.arange(8., 50., 0.5) * u.Angstrom
energies = wave.to(u.keV, equivalencies=u.spectral()).value
outpath = '../../../Dropbox/ARCUS/rays/semi-compact'

mypointing = FixedPointing(coords=(30, 30.))

for i, e in enumerate(energies):
    print '{0}/{1}'.format(i + 1, len(energies))
    mysource = PointSource((30., 30.), energy=e, flux=1.)

    photons = mysource.generate_photons(n_photons)
    photons = mypointing(photons)
    photons = arcus.arcus_extra_det(photons)

    photonsm = mysource.generate_photons(n_photons)
    photonsm = mypointing(photonsm)
    photonsm = arcus.arcus_extra_det_m(photonsm)
    photonsm['aperture'] += 2

    with enable_merge_strategies(utils.MergeIdentical):
        out = table.vstack([photons, photonsm])

    out.write(os.path.join(outpath, 'wave{0:05.2f}.fits'.format(wave.value[i])),
              overwrite=True)
