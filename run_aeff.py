import numpy as np

from marxs.source import PointSource, FixedPointing

from astropy.table import Table
import astropy.units as u
import arcus

n_photons = 1e4
wave = np.arange(8., 60., 0.5) * u.Angstrom
#energies = np.arange(.2, 1.9, .01)
energies = wave.to(u.keV, equivalencies=u.spectral()).value
outfile = '../results/aeff.fits'

out_e = []
out_aeff = []

try:
    for i, e in enumerate(energies):
        mysource = PointSource((30., 30.), energy=e, flux=1.)
        photons = mysource.generate_photons(n_photons)

        mypointing = FixedPointing(coords=(30, 30.))
        photons = mypointing(photons)
        photons = arcus.arcus(photons)
        print '{0}/{1}'.format(i + 1, len(energies))
        out_e.append(e)
        # keep only those photons that went through a grating facet
        # The other will be absorbed by support structure.
        photons = photons[photons['facet'] >= 0]
        # select detected photons
        detected = photons['CCD_ID'] >= 0.
        bincount = np.bincount(np.asarray((photons['order'][detected] + 20), dtype=int),
                               weights=photons['probability'][detected],
                               minlength=15)
        out_aeff.append(bincount / n_photons)

finally:
    # Any exception in the calculation - save what we have so far!
    tab = Table([out_e, out_aeff],
                names=('energy', 'fA'))
    tab.meta['nphotons'] = n_photons
    tab.write(outfile, overwrite=True)
