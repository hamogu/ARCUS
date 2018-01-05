import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import astropy.units as u
from scipy.interpolate import RectBivariateSpline
from numpy.core.umath_tests import inner1d

from marxs.optics.aperture import RectangleAperture, MultiAperture
from marxs.optics import (PerfectLens, GlobalEnergyFilter,
                          RadialMirrorScatter)
from marxs.simulator import Parallel
from marxs.math.utils import e2h, h2e, norm_vector
from marxs.math.polarization import parallel_transport

from .load_csv import load_table2d, load_number, load_table
from .constants import xyz2zxy

inplanescatter = 10. / 2.3545 / 3600 / 180. * np.pi
perpplanescatter = 1.5 / 2.345 / 3600. / 180. * np.pi

focallength = 12000.


spogeom = load_table('spos', 'petallayout')
spogeom['r_mid'] = (spogeom['outer_radius'] + spogeom['inner_radius']) / 2
spo_pos4d = []
# Convert angle to quantity here to make sure that unit is taken into account
for row, ang in zip(spogeom, u.Quantity(spogeom['clocking_angle']).to(u.rad).value):
    spo_pos4d.append(compose([0,  # focallength,  # - spogeom[i]['d_from_12m']
                              row['r_mid'] * np.sin(ang),
                              row['r_mid'] * np.cos(ang)],
                             euler2mat(-ang, 0., 0.),
                             # In detail this should be (primary_length + gap + secondary_length) / 2
                             # but the gap is somewhat complicated and this is only used
                             # for display, we'll ignore that for now.
                             [row['primary_length'],
                              row['azwidth'] / 2.,
                              (row['outer_radius'] - row['inner_radius']) / 2.]))

spo_pos4d = [np.dot(xyz2zxy, s) for s in spo_pos4d]

reflectivity = load_table2d('spos', 'reflectivity')
reflectivity_interpolator = RectBivariateSpline(reflectivity[0].data,
                                                reflectivity[1].data,
                                                reflectivity[3][0])


class PerfectLensSegment(PerfectLens):
    def __init__(self, **kwargs):
        self.d_center_optax = kwargs.pop('d_center_optical_axis')
        super(PerfectLensSegment, self).__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        # A ray through the center is not broken.
        # So, find out where a central ray would go.
        p_opt_axis = self.geometry('center') - self.d_center_optax * self.geometry('e_z')
        focuspoints = h2e(p_opt_axis) + self.focallength * norm_vector(h2e(photons['dir'][intersect]))
        dir = norm_vector(e2h(focuspoints - h2e(interpos[intersect]), 0))
        pol = parallel_transport(photons['dir'].data[intersect, :], dir,
                                 photons['polarization'].data[intersect, :])
        angle = np.arccos(np.abs(inner1d(h2e(dir),
                                         norm_vector(h2e(photons['dir'][intersect])))))
        return {'dir': dir, 'polarization': pol,
                'probability': reflectivity_interpolator(photons['energy'][intersect],
                                                         np.rad2deg(angle) / 2,
                                                         grid=False)**2
                }


class SPOChannelMirror(Parallel):
    def __init__(self, **kwargs):
        kwargs['elem_pos'] = spo_pos4d
        kwargs['elem_class'] = PerfectLensSegment
        kwargs['elem_args'] = {'d_center_optical_axis': list(spogeom['r_mid']),
                               'focallength': list(spogeom['focal_length'])}
        kwargs['id_col'] = 'xou'
        super(SPOChannelMirror, self).__init__(**kwargs)


class SPOChannelasAperture(MultiAperture):
    def __init__(self, **kwargs):
        elements = [RectangleAperture(pos4d) for pos4d in spo_pos4d]
        kwargs['elements'] = elements
        super(SPOChannelasAperture, self).__init__(**kwargs)
        for e in self.elements:
            e.pos4d = np.dot(self.pos4d, e.pos4d)

geometricopening = load_number('spos', 'geometricthroughput', 'transmission')
geometricthroughput = GlobalEnergyFilter(filterfunc=lambda e: geometricopening,
                                         name='SPOgeometricthrougput')


class ScatterPerChannel(RadialMirrorScatter):
    '''A scatter of infinite size that identifies photons by spo id

    This bypasses the intersection calculation and instead
    just selects photons for this scatter by spo id.

    Parameters
    ----------
    min_id : integer
        Photons with spo id between ``min_id`` and ``min_id + 1000`` are
        scattered.
    '''
    display = {'shape': 'None'}

    def __init__(self, **kwargs):
        self.min_id = kwargs.pop('min_id')
        super(ScatterPerChannel, self).__init__(**kwargs)

    def __call__(self, photons):
        intersect = ((photons['xou'] >= self.min_id) &
                     (photons['xou'] < (self.min_id + 1000)))
        # interpos and intercoos is used to automatically set new position
        # (which we want unaltered, thus we pass pos) and local coords
        # (which we don't care about, thus we pass zeroth in the right shape.
        return self.process_photons(photons, intersect, photons['pos'].data,
                                    np.zeros((len(photons), 2)))
