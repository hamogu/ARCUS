import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import astropy.units as u
from scipy.interpolate import interp1d

from marxs.optics.aperture import RectangleAperture, MultiAperture
from marxs.optics import (PerfectLens, GlobalEnergyFilter,
                          RadialMirrorScatter)
from marxs.simulator import Parallel
from marxs.math.utils import e2h, h2e, norm_vector
from marxs.math.polarization import parallel_transport

from .load_csv import load_table, load_number

inplanescatter = 10. / 2.3545 / 3600 / 180. * np.pi
perpplanescatter = 1.5 / 2.345 / 3600. / 180. * np.pi

focallength = 12000.


spogeom = load_table('spos', 'petallayout')
spo_pos4d = []
# Convert angle to quantity here to make sure that unit is taken into account
for row, ang in zip(spogeom, u.Quantity(spogeom['angle']).to(u.rad).value):
    spo_pos4d.append(compose([0,  # focallength,  # - spogeom[i]['d_from_12m']
                              row['r_mid'] * np.sin(ang),
                              row['r_mid'] * np.cos(ang)],
                             euler2mat(-ang, 0., 0.),
                             [row['height'] / 2.,
                              row['width'] / 2.,
                              row['depth'] / 2.]))


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
        kwargs['elem_args'] = {'d_center_optical_axis': list(spogeom['r_mid']),
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

geometricopening = load_number('spos', 'geometricthroughput', 'transmission')
geometricthroughput = GlobalEnergyFilter(filterfunc=lambda e: geometricopening,
                                         name='SPOgeometricthrougput')


def get_reflectivityfilter():
    tab = load_table('spos', 'reflectivity_simple')
    en = tab['energy'].to(u.keV, equivalencies=u.spectral())
    return GlobalEnergyFilter(filterfunc=interp1d(en, tab['reflectivity']**2),
                              name='double relectivity')
doublereflectivity = get_reflectivityfilter()


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
        intersect = ((photons['spo'] > self.min_id) &
                     (photons['spo'] < (self.min_id + 1000)))
        # interpos and intercoos is used to automatically set new position
        # (which we want unaltered, thus we pass pos) and local coords
        # (which we don't care about, thus we pass zeroth in the right shape.
        return self.process_photons(photons, intersect, photons['pos'].data,
                                    np.zeros((len(photons), 2)))
