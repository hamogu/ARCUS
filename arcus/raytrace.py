import os
import numpy as np
import copy

import marxs
import marxs.optics
from marxs.simulator import Sequence
from marxs.source import PointSource, FixedPointing
from marxs.analysis import resolvingpower_per_order, weighted_per_order

from mayavi import mlab
from marxs.math.pluecker import h2e
import marxs.visualization.mayavi


from arcus.arcus import rowland, aper, mirror, gas, catsupport, det

# Place an additional detector in the focal plane for comparison
# Detectors are transparent to allow this stuff
detfp = marxs.optics.FlatDetector(zoom=[.2, 1000, 1000])
detfp.loc_coos_name = ['detfp_x', 'detfp_y']
detfp.detpix_name = ['detfppix_x', 'detfppix_y']
detfp.display['opacity'] = 0.2

# Place an additional detector on the Rowland circle.
detcirc = marxs.optics.CircularDetector.from_rowland(rowland, width=20)
detcirc.loc_coos_name = ['detcirc_phi', 'detcirc_y']
detcirc.detpix_name = ['detcircpix_x', 'detcircpix_y']
detcirc.display['opacity'] = 0.2


uptomirror = Sequence(elements=[aper, mirror])


keeppos = marxs.simulator.KeepCol('pos')
mission = Sequence(elements=[aper, mirror, gas, catsupport, det, detfp],
                   postprocess_steps=[keeppos])

star = PointSource(coords=(23., 45.), flux=5.)
pointing = FixedPointing(coords=(23., 45.))
photons = star.generate_photons(exposuretime=2000)
photons = pointing(photons)

### Look at different energies for some orders in detail
p = uptomirror(photons.copy())

gratings = copy.deepcopy(gas)
p1 = p.copy()
p02 = p.copy()
p02['energy'] = 0.2

gratingeff = marxs.optics.constant_order_factory(0)
for elem in gratings.elements:
            elem.order_selector = gratingeff

p1o0 = gratings(p1.copy())
p02o0 = gratings(p02.copy())

gratingeff = marxs.optics.constant_order_factory(-5)
for elem in gratings.elements:
            elem.order_selector = gratingeff

p1o2 = gratings(p1.copy())
p02o2 = gratings(p02.copy())

gratingeff = marxs.optics.constant_order_factory(-10)
for elem in gratings.elements:
            elem.order_selector = gratingeff

p1o4 = gratings(p1.copy())
p02o4 = gratings(p02.copy())

for ptemp in [p1o0, p1o2, p1o4, p02o0, p02o2, p02o4]:
    ptemp.remove_rows(np.isnan(ptemp['order']))
    ptemp = detcirc(ptemp)


fig = mlab.figure()
mlab.clf()

d = np.dstack(keeppos2.data)
d = np.swapaxes(d, 1, 2)
d = h2e(d)

marxs.visualization.mayavi.plot_rays(d, viewer=fig)


### Main flow continues here
p = mission(photons.copy())


fig = mlab.figure()
mlab.clf()

d = np.dstack(keeppos.data)
d = np.swapaxes(d, 1, 2)
d = h2e(d)

marxs.visualization.mayavi.plot_rays(d, scalar=p['order'], viewer=fig)
mission.plot(format="mayavi", viewer=fig)

theta, phi = np.mgrid[-0.2 + np.pi:0.2 + np.pi:60j, -1:1:60j]
rowland.plot(theta=theta, phi=phi, viewer=fig, format='mayavi')

# Ryan's rays
from astropy.io import fits
import marxs.utils
from marxs.design import GratingArrayStructure
from arcus import CATGrating, order_selector, blazemat

ryan = fits.getdata(os.path.join('..', '160418_SPOTrace.fits'))
rp = marxs.utils.generate_test_photons(ryan.shape[1])
for col, indoffset in zip(['pos', 'dir'], [0, 3]):
    rp[col][:, 0] = ryan[2 + indoffset, :]
    rp[col][:, 1] = ryan[1 + indoffset, :]
    rp[col][:, 2] = ryan[0 + indoffset, :]


gas2 = GratingArrayStructure(rowland=rowland, d_element=30.,
                             x_range=[1e4, 1.4e4],
                             radius=[300, 800], phi=[-0.3 + np.pi / 2, .3 + np.pi / 2],
                             elem_class=CATGrating,
                             elem_args={'d': 2e-4, 'zoom': [1., 10., 10.],
                                        'orientation': blazemat,
                                        'order_selector': order_selector})
keeppos2 = marxs.simulator.KeepCol('pos')
arcusshort = Sequence(elements=[gas2, catsupport, det, detfp],
                      postprocess_steps=[keeppos2])

rp1 = arcusshort(rp.copy())

d = np.dstack(keeppos2.data)
d = np.swapaxes(d, 1, 2)
d = h2e(d)

marxs.visualization.mayavi.plot_rays(d, viewer=fig)
arcusshort.plot(format="mayavi", viewer=fig)

theta, phi = np.mgrid[-0.2 + np.pi:0.2 + np.pi:60j, -1:1:60j]
rowland.plot(theta=theta, phi=phi, viewer=fig, format='mayavi')
