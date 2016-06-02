import numpy as np
import marxs
import marxs.optics
from marxs.simulator import Sequence
from marxs.source import PointSource, FixedPointing
from marxs.analysis import fwhm_per_order, weighted_per_order

from mayavi import mlab
from marxs.math.pluecker import h2e
import marxs.visualization.mayavi


from arcus import rowland, aper, mirror, gas, catsupport, det

# Place an additional detector in the focal plane for comparison
# Detectors are transparent to allow this stuff
detfp = marxs.optics.FlatDetector(zoom=[.2, 1000, 1000])
detfp.loc_coos_name = ['detfp_x', 'detfp_y']
detfp.detpix_name = ['detfppix_x', 'detfppix_y']
detfp.display['opacity'] = 0.2


keeppos = marxs.simulator.KeepCol('pos')
arcus = Sequence(elements=[aper, mirror, gas, catsupport, det, detfp],
                 postprocess_steps=[keeppos])

star = PointSource(coords=(23., 45.), flux=5.)
pointing = FixedPointing(coords=(23., 45.))
photons = star.generate_photons(exposuretime=200)
p = pointing(photons)
p = arcus(p)


fig = mlab.figure()
mlab.clf()

d = np.dstack(keeppos.data)
d = np.swapaxes(d, 1, 2)
d = h2e(d)

marxs.visualization.mayavi.plot_rays(d, scalar=p['energy'], viewer=fig)
arcus.plot(format="mayavi", viewer=fig)

theta, phi = np.mgrid[-0.2 + np.pi:0.2 + np.pi:60j, -1:1:60j]
rowland.plot(theta=theta, phi=phi, viewer=fig, format='mayavi')
