import numpy as np

from marxs.source import PointSource, FixedPointing
import marxs
import astropy.units as u
import arcus

from mayavi import mlab
from marxs.math.pluecker import h2e
import marxs.visualization.mayavi


n_photons = 1e4
wave = np.arange(8., 50., 0.5) * u.Angstrom
#energies = np.arange(.2, 1.9, .01)
energies = wave.to(u.keV, equivalencies=u.spectral()).value
outfile = '../results/aeff.fits'

e = 0.5

mysource = PointSource((30., 30.), energy=e, flux=1.)
photons = mysource.generate_photons(n_photons)

mypointing = FixedPointing(coords=(30, 30.))
photons = mypointing(photons)

keeppos = marxs.simulator.KeepCol('pos')
arcus.arcus_extra_det.postprocess_steps = [keeppos]

photons = arcus.arcus_extra_det(photons)


fig = mlab.figure()
mlab.clf()

d = np.dstack(keeppos.data)
d = np.swapaxes(d, 1, 2)
d = h2e(d)

marxs.visualization.mayavi.plot_rays(d, scalar=photons['order'], viewer=fig)
arcus.arcus.plot(format="mayavi", viewer=fig)

theta, phi = np.mgrid[-0.2 + np.pi:0.2 + np.pi:60j, -1:1:60j]
arcus.rowland.plot(theta=theta, phi=phi, viewer=fig, format='mayavi')


photonsm = mysource.generate_photons(n_photons)
photonsm = mypointing(photonsm)

keepposm = marxs.simulator.KeepCol('pos')
arcus.arcus_extra_det_m.postprocess_steps = [keepposm]

photonsm = arcus.arcus_extra_det_m(photonsm)
d = np.dstack(keepposm.data)
d = np.swapaxes(d, 1, 2)
d = h2e(d)

marxs.visualization.mayavi.plot_rays(d, scalar=photonsm['order'], viewer=fig)
arcus.arcusm.plot(format="mayavi", viewer=fig)


from astropy.stats import sigma_clipped_stats
ind = np.isfinite(photons['order']) & (photons['CCD_ID'] >=0)
