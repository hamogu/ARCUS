import numpy as np

from marxs.source import PointSource, FixedPointing

from astropy.table import Table
from arcus import arcus

n_photons = 1e4
energies = np.arange(.2, 1.9, .01)
outfile = '../results/aeff.fits'

out_e = []
out_aeffg = []
out_aeff0 = []

try:
    i = 0
    for e in energies:
        mysource = PointSource((30., 30.), energy=e, flux=1.)
        photons = mysource.generate_photons(n_photons)

        mypointing = FixedPointing(coords=(30, 30.))
        photons = mypointing(photons)
        photons = arcus(photons)
        print '{0}/{1}'.format(i, len(energies))
        i = i + 1
        out_e.append(e)
        # select detected photons
        detected = photons['CCD_ID'] >=0.
        diffracted = photons['order'] < 0.
        o0 = photons['order'] == 0.
        out_aeffg.append(photons['probability'][detected & diffracted].sum() / n_photons)
        out_aeff0.append(photons['probability'][detected & o0].sum() / n_photons)

finally:
    # Any exception in the calculation - save what we have so far!
    tab = Table([out_e, out_aeff0, out_aeffg],
                names=('energy', 'fA0', 'fAg'))
    tab.meta['nphotons'] = n_photons
    tab.write(outfile, overwrite=True)
