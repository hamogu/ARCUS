import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
from astropy.table import Table

from marxs.optics.aperture import RectangleAperture, MultiAperture
from marxs.optics import RadialMirrorScatter, PerfectLens, FlatStack
from marxs.simulator import Parallel
from marxs.math.utils import e2h, h2e, norm_vector
from marxs.math.polarization import parallel_transport


inplanescatter=10. / 2.3545 / 3600 / 180. * np.pi
perpplanescatter=1.5 / 2.345 / 3600. / 180. * np.pi

focallength = 12000.

spo_geometry = '''
r_mid   height  depth  width  d_from_12m
348.343 167.926 55.800 50.159 5.053
410.538 142.830 55.800 49.839 7.016
472.733 124.363 55.800 49.614 9.301
534.927 110.571 55.800 89.363 11.905
597.122  99.415 55.800 82.476 14.829
656.317  90.391 55.800 77.572 17.908
721.512  82.951 55.800 86.892 21.632
783.707  76.710 55.800 82.053 25.510
'''

spo_angles = [
    [-27.653, -16.592, -5.531, 5.531, 16.592, 27.653],
    [-23.174, -13.904, -4.635, 4.635, 13.904, 23.174],
    [-19.942, -11.965, -3.988, 3.988, 11.965, 19.942],
    [-16.065, -5.355, 5.355, 16.065],
    [-14.317, -4.772, 4.772, 14.317],
    [-12.972, -4.324, 4.324, 12.972],
    [-11.756, -3.919, 3.919, 11.756],
    [-10.791, -3.597, 3.597, 10.791]]

spo_pos4d = []
spo_rmid = []
spogeom = Table.read(spo_geometry, format='ascii')
for i, row in enumerate(spo_angles):
    for ang in row:
        spo_pos4d.append(compose([0, # focallength,  # - spogeom[i]['d_from_12m']
                                  spogeom[i]['r_mid'] * np.sin(np.deg2rad(ang)),
                                  spogeom[i]['r_mid'] * np.cos(np.deg2rad(ang))],
                                 euler2mat(-np.deg2rad(ang), 0., 0.),
                                 [spogeom[i]['height'] / 2.,
                                  spogeom[i]['width'] / 2.,
                                  spogeom[i]['depth'] / 2.]
                                 )
                         )
        spo_rmid.append(spogeom[i]['r_mid'])


class PerfectLensSegment(PerfectLens):
    def __init__(self, **kwargs):
        self.d_center_optax = kwargs.pop('d_center_optical_axis')
        super(PerfectLensSegment, self).__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        # A ray through the center is not broken.
        # So, find out where a central ray would go.
        p_opt_axis = self.geometry('center') - self.d_center_optax * self.geometry('e_z')
        focuspoints = h2e(p_opt_axis) + self.focallength * norm_vector(h2e(photons['dir'][intersect]))
        dir = e2h(focuspoints - h2e(interpos[intersect]), 0)
        pol = parallel_transport(photons['dir'].data[intersect, :], dir,
                                 photons['polarization'].data[intersect, :])
        return {'dir': dir, 'polarization': pol}


class SPOChannelMirror(Parallel):
    def __init__(self, **kwargs):
        kwargs['elem_pos'] = spo_pos4d
        kwargs['elem_class'] = PerfectLensSegment
        kwargs['elem_args'] = {'d_center_optical_axis': spo_rmid,
                               'focallength': focallength}
        kwargs['id_col'] = 'spo'
        super(SPOChannelMirror, self).__init__(**kwargs)


class SPOChannelasAperture(MultiAperture):
    def __init__(self, **kwargs):
        elements = [RectangleAperture(pos4d) for pos4d in spo_pos4d]
        kwargs['elements'] = elements
        super(SPOChannelasAperture, self).__init__(**kwargs)
        for e in self.elements:
            e.pos4d = np.dot(self.pos4d, e.pos4d)
