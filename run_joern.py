import os
import numpy as np

from marxs.source import PointSource, FixedPointing

from astropy.table import Table
import astropy.units as u
from astropy.io import fits
import arcus

n_photons = 1e4

wave = np.arange(8., 50., 0.5) * u.Angstrom
energies = wave.to(u.keV, equivalencies=u.spectral()).value

outdir = '/melkor/d1/guenther/Dropbox/ARCUS/raysforJoern/'

pointing_offsets = [0, np.deg2rad(0.1)]


def swapxz(arr):
    temp = np.zeros_like(arr)
    temp[:, 0] = arr[:, 2]
    temp[:, 1] = arr[:, 1]
    temp[:, 2] = arr[:, 0]
    temp[:, 3] = arr[:, 3]
    return temp


def write_joerntables(photons, outdir, ie, indx, offx, indy, offy):
    if not np.all(photons['energy'] == photons['energy'][0]):
        raise ValueError("need monoenergetic simulations")
    orders = set(photons['order'][np.isfinite(photons['order'])])
    for o in orders:
        filename = os.path.join(outdir, '{0}_{1}_{2}_{3}.fits'.format(ie, indx,indy, int(np.abs(o))))
        ind = (photons['order'] == o)
        tab = Table()
        tab['pos'] = swapxz(photons['pos'][ind]) * u.mm
        tab['dir'] = swapxz(photons['dir'][ind]) * u.mm
        tab['time'] = photons['time'][ind] * u.s
        tab.write(filename, overwrite=True)
        # easier to open again to add keywords then use
        # fits interface directly above
        hdulist = fits.open(filename, mode='update')
        hdulist[0].header['ENERGY'] = (photons[0]['energy'], 'energy in keV')
        hdulist[0].header['ORDER'] = (o, 'diffraction order')
        hdulist[0].header['OFFX'] = (offx, 'offset from optical axis in radian')
        hdulist[0].header['OFFY'] = (offy, 'offset from optical axis in radian')
        hdulist[0].header['N_PHOTONS'] = (n_photons, 'Number of photons per simulation')
        hdulist.close()

for ix, offx in enumerate(pointing_offsets):
    for iy, offy in enumerate(pointing_offsets):
        for ie, e in enumerate(energies):
            print ix, iy, ie
            mysource = PointSource((0., 0.), energy=e, flux=1.)
            photons = mysource.generate_photons(n_photons)

            mypointing = FixedPointing(coords=(np.rad2deg(offx),
                                               np.rad2deg(offy)))
            photons = mypointing(photons)
            photons = arcus.arcus_joern(photons)
            write_joerntables(photons, outdir, ie, ix, offx, iy, offy)
