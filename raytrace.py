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

# Ryan's rays
from astropy.io import fits
import marxs.utils
from marxs.design import GratingArrayStructure
from arcus import CATGrating, order_selector, blazemat

ryan = fits.getdata(os.path.join(path, '..', '160418_SPOTrace.fits'))
rp = marxs.utils.generate_test_photons(ryan.shape[1])
for col, indoffset in zip(['pos', 'dir'], [0, 3]):
    rp[col][:, 0] = ryan[2 + indoffset, :]
    rp[col][:, 1] = ryan[1 + indoffset, :]
    rp[col][:, 2] = ryan[0 + indoffset, :]


gas2 = GratingArrayStructure(rowland=rowland, d_element=30.,
                             x_range=[1e4, 1.4e4],
                             radius=[300, 800], phi=[-0.3+np.pi/2, .3+np.pi/2],
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

orders = np.arange(-10, 0)
fwhm_per_order(gas2, rp.copy(), orders)


    if len(orders) != data.shape[0]:
        raise ValueError('First dimension of "data" must match length of "orders".')
    if len(energy) != data.shape[1]:
        raise ValueError('Second dimension of "data" must match length of "energy".')

    weights = np.zeros_like(data)
    for i, o in enumerate(orders):
        ind_o = (gratingeff.orders == o).nonzero()[0]
        if len(ind_o) != 1:
            raise KeyError('No data for order {0} in gratingeff'.format(o))
        en_sort = np.argsort(gratingeff.energy)
        weights[o, :] = np.interp(energy, gratingeff.energy[en_sort],
                                  gratingeff.prob[:, ind_o[0]][en_sort])

    return np.ma.average(data, axis=0, weights=weights)
