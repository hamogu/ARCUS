from astropy.table import Table
import astropy.table
from astropy.utils.metadata import enable_merge_strategies
from marxs import utils

from marxs.source import PointSource, JitterPointing
import arcus

n_photons = 1e7

EQPegAspec = Table.read('../inputdata/EQPegA_flux.tbl', format='ascii',
                        names=['energy', 'flux'])
# restrict table to ARCUS energy range
EQPegAspec = EQPegAspec[(EQPegAspec['energy'] > 0.25) & (EQPegAspec['energy'] < 1.5)]

coord = astropy.coordinates.SkyCoord.from_name("EQ Peg A")


# define position and spectrum of source
mysource = PointSource((coord.ra.value, coord.dec.value), energy=EQPegAspec, flux=1.)
mypointing = JitterPointing(coords=(coord.ra.value, coord.dec.value), jitter=arcus.jitter_sigma)

# MARXS code is still missing the implementation to rescale the input spectrum to the
# effective area of the instrument.
# Thus, set total number of photons to simulate here by hand.
photons = mysource.generate_photons(n_photons / 2)
photons = mypointing(photons)
photons = arcus.arcus_extra_det(photons)

photonsm = mysource.generate_photons(n_photons / 2)
photonsm = mypointing(photonsm)
photonsm = arcus.arcus_extra_det_m(photonsm)


with enable_merge_strategies(utils.MergeIdentical):
    allphot = astropy.table.vstack([photons, photonsm])

allphot.write('/melkor/d1/guenther/Dropbox/ARCUS/rays/EQPegA1e7.fits', overwrite=True)
