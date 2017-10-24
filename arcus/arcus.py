from copy import deepcopy
import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d
import transforms3d

import marxs
from marxs.simulator import Sequence, KeepCol
from marxs.optics import (GlobalEnergyFilter,
                          FlatDetector, CATGrating,
                          RadialMirrorScatter)
from marxs import optics
from marxs.design.rowland import RowlandCircleArray
import marxs.analysis

from .ralfgrating import (InterpolateRalfTable, RalfQualityFactor,
                          catsupportbars, catsupport,
                          RectangularGrid)
from . import spo
from . import boom
from .load_csv import load_number, load_table
from .utils import tagversion
from .constants import xyz2zxy
from .generate_rowland import make_rowland_from_d_BF_R_f

__all__ = ['xyz2zxy',
           'jitter_sigma',
           'Arcus', 'ArcusForPlot', 'ArcusForSIXTE']

jitter_sigma = load_number('other', 'pointingjitter',
                           'FWHM') / 2.3545
defaultconf = make_rowland_from_d_BF_R_f(600., 5900.)
defaultconf['blazeang'] = 1.8
defaultconf['n_CCDs'] = 16
defaultconf['phi_det_start'] = 0.04297

id_num_offset = {'1': 0,
                 '2': 1000,
                 '1m': 10000,
                 '2m': 11000}


class Aperture(optics.MultiAperture):
    def __init__(self, conf, channels=['1', '2', '1m', '2m'], **kwargs):
        # Set a little above entrance pos (the mirror) for display purposes.
        # Thus, needs to be geometrically bigger for off-axis sources.
        rect1 = optics.RectangleAperture(position=[0, 550, 12200],
                                         zoom=[1, 220, 330],
                                         orientation=xyz2zxy[:3, :3])
        rect2 = optics.RectangleAperture(position=[0, -550, 12200],
                                         zoom=[1, 220, 330],
                                         orientation=xyz2zxy[:3, :3])

        rect1m = optics.RectangleAperture(pos4d=np.dot(conf['shift_optical_axis_12'], rect1.pos4d))
        rect2m = optics.RectangleAperture(pos4d=np.dot(conf['shift_optical_axis_12'], rect2.pos4d))
        apers = []

        for a, b in zip(['1', '2', '1m', '2m'], [rect1, rect2, rect1m, rect2m]):
            if a in channels:
                apers.append(b)
        super(Aperture, self).__init__(elements=apers, **kwargs)
        # Prevent aperture edges from overlapping in plot
        self.elements[0].display['outer_factor'] = 2


def spomounting(photons):
    '''Remove photons that do not go through an SPO but hit the
    frame part of the petal.'''
    photons['probability'][photons['spo'] < 0] = 0.
    return photons


class SimpleSPOs(Sequence):

    def __init__(self, conf, channels=['1', '2', '1m', '2m'],
                 inplanescatter=10. / 2.3545 / 3600 / 180. * np.pi,
                 perpplanescatter=1.5 / 2.345 / 3600. / 180. * np.pi,
                 **kwargs):
        entrancepos = np.array([0., -conf['offset_spectra'], 12000.])
        entranceposm = np.array([2. * conf['d'], +conf['offset_spectra'], 12000.])
        rot180 = transforms3d.euler.euler2mat(np.pi, 0,0,'szyx')
        # Make lens a little larger than aperture, otherwise an non on-axis ray
        # (from pointing jitter or an off-axis source) might miss the mirror.
        mirror = []
        for chan in channels:
            entrancepos = conf['pos_opt_ax'][chan][:3]
            entrancepos[2] += 12000
            rot = np.eye(3) if chan in ['1', '1m'] else rot180
            mirror.append(spo.SPOChannelMirror(position=entrancepos,
                                               orientation=np.dot(rot, xyz2zxy[:3, :3]),
                                               id_num_offset=id_num_offset[chan]))
        # Get entracepos fopr the the follo9wiung two statements
        if ('1' in channels) or ('2' in channels):
            mirror.append(RadialMirrorScatter(inplanescatter=inplanescatter,
                                              perpplanescatter=perpplanescatter,
                                              position=entrancepos, zoom=[1, 220, 900],
                                              orientation=xyz2zxy[:3, :3]))
        if ('1m' in channels) or ('2m' in channels):
            mirror.append(RadialMirrorScatter(inplanescatter=inplanescatter,
                                              perpplanescatter=perpplanescatter,
                                              position=entranceposm, zoom=[1, 220, 900],
                                              orientation=xyz2zxy[:3, :3]))
        mirror.append(spo.geometricthroughput)
        mirror.append(spo.doublereflectivity)
        mirror.append(spomounting)
        super(SimpleSPOs, self).__init__(elements=mirror, **kwargs)


class CATGratings(Sequence):
    order_selector_class = InterpolateRalfTable
    gratquality_class = RalfQualityFactor
    grid_width_x = 180
    grid_width_y = 300

    def __init__(self, conf, channels=['1', '2', '1m', '2m'], **kwargs):

        elements = []

        self.order_selector = self.order_selector_class()
        self.gratquality = self.gratquality_class()
        blazemat = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]),
                                                          np.deg2rad(-conf['blazeang']))
        blazematm = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]),
                                                           np.deg2rad(conf['blazeang']))

        gratinggrid = {'rowland': conf['rowland'],
                       'd_element': 32., 'z_range': [1e4, 1.4e4],
                       'elem_class': CATGrating,
                       'elem_args': {'d': 2e-4, 'zoom': [1., 13.5, 13.],
                                     'orientation': blazemat,
                                     'order_selector': self.order_selector},
                       'parallel_spec': np.array([1., 0., 0., 0.])
                       }
        for chan in channels:
            gratinggrid['rowland'] = conf['rowland_'] + chan
            b = blazematm if 'm' in chan else blazemat
            gratinggrid['elem_args']['orientation'] = b
            gratinggrid['normal_spec'] = conf['pos_opt_ax'][chan]
            xm, ym = conf['pos_opt_ax'][chan][:2]
            sig = 1 if chan in ['1', '2'] else -1
            x_range = [-self.grid_width_x + xm,
                       +self.grid_width_y + xm]
            y_range = [sig * (600 - ym - self.grid_width_y),
                       sig * (600 - ym + self.grid_width_y)]
            y_range.sort()
            elements.append(RectangularGrid(x_range=x_range, y_range=y_range,
                                            id_num_offset=self.id_num_offset[chan],
                                            **gratinggrid))
        elements.extend([catsupport, catsupportbars, self.gratquality])
        super(CATGratings, self).__init__(elements=elements, **kwargs)


class FiltersAndQE(Sequence):

    filterlist = [('filters', 'sifilter'),
                  ('filters', 'opticalblocking'),
                  ('filters', 'uvblocking'),
                  ('detectors', 'contam'),
                  ('detectors', 'qe')]

    def get_filter(self, directory, name):
        tab = load_table(directory, name)
        en = tab['energy'].to(u.keV, equivalencies=u.spectral())
        return GlobalEnergyFilter(filterfunc=interp1d(en, tab[tab.colnames[1]]),
                                  name=name)

    def __init__(self, conf, channels, **kwargs):
        elems = [self.get_filter(*n) for n in self.filterlist]
        super(FiltersAndQE, self).__init__(elements=elems, **kwargs)


class DetMany(RowlandCircleArray):
    elem_class = FlatDetector
    elem_args = {'pixsize': 0.024, 'zoom': [1, 24.576, 12.288]}
    d_element = elem_args['zoom'][1] * 2 + 0.5  # 500 mu gap between detectors
    theta = [np.pi - 0.5, np.pi + 0.5]

    def __init__(self, conf, **kwargs):
        super(DetMany, self).__init__(rowland=conf['rowland_central'],
                                    elem_class=self.elem_class,
                                    elem_args=self.elem_args,
                                    d_element=self.d_element,
                                    theta=self.theta)


class Det16(DetMany):
    '''Place only hand-selected 16 CCDs'''
    def __init__(self, conf, theta=[3.1255, 3.1853, 3.2416, 3.301],  **kwargs):
        self.theta=theta
        super(Det16, self).__init__(conf, **kwargs)
        assert len(self.elements) == 16
        # but make real detectors orange
        disp = deepcopy(self.elements[0].display)
        disp['color'] = 'orange'
        for e in self.elements:
            e.display = disp


class DetTwoStrips(DetMany):
    offset_strip = 0.1
    'Offset for one strip to make chip gaps different in CCD lengths'

    def __init__(self, conf, **kwargs):
        r = conf['rowland_central'].r
        phi_m = np.arcsin(conf['d'] / r) + np.pi
        angle_strip = conf['n_CCDs'] / 2 * self.d_element / r
        p0 = conf['phi_det_start']
        offset = self.offset_strip * self.d_element / r
        # +- 1e4 at the boundaries, otherwise rounding error can already
        # and an entire extra CCD
        self.theta = [phi_m - p0 - angle_strip + 1e-4,
                      phi_m - p0,
                      phi_m + p0 + offset,
                      phi_m + p0 + angle_strip + offset - 1e-4]
        super(DetTwoStrips, self).__init__(conf, **kwargs)
        assert len(self.elements) == conf['n_CCDs']
        # but make real detectors orange
        disp = deepcopy(self.elements[0].display)
        disp['color'] = 'orange'
        for e in self.elements:
            e.display = disp


class CircularDetector(marxs.optics.CircularDetector):
    def __init__(self, rowland, name, width=20, **kwargs):
        # Step 1: Get position and size from Rowland torus
        pos4d_circ = transforms3d.affines.compose([rowland.R, 0, 0],
                                                  np.eye(3),
                                                  [rowland.r, rowland.r, width])
        # Step 2: Transform to global coordinate system
        pos4d_circ = np.dot(rowland.pos4d, pos4d_circ)
        # Step 3: Make detector
        super(CircularDetector, self).__init__(pos4d=pos4d_circ, phi_offset=-np.pi)
        self.loc_coos_name = [name + '_phi', name + '_y']
        self.detpix_name = [name + 'pix_x', name + 'pix_y']
        self.display = deepcopy(self.display)
        self.display['opacity'] = 0.1


class FocalPlaneDet(marxs.optics.FlatDetector):
    loc_coos_name = ['detfp_x', 'detfp_y']
    detpix_name = ['detfppix_x', 'detfppix_y']

    def __init__(self, **kwargs):
        if ('zoom' not in kwargs) and ('pos4d' not in kwargs):
            kwargs['zoom'] = [.2, 10000, 10000]
        if ('orientation' not in kwargs) and ('pos4d' not in kwargs):
            kwargs['orientation'] = xyz2zxy[:3, :3]
        super(FocalPlaneDet, self).__init__(**kwargs)


class Arcus(Sequence):

    aper_class = Aperture
    spo_class = SimpleSPOs
    gratings_class = CATGratings
    filter_and_qe_class = FiltersAndQE
    '''Set any of these classes to None to not have them included.
    (e.g. SIXTE does filters and QE itself).
    '''

    def add_boom(self, conf):
        '''Add four sided boom. Only the top two bays contribute any
        absorption, so we can save time by not modelling the remaining bays.'''
        return [boom.FourSidedBoom(orientation=xyz2zxy[:3, :3],
                                   position=[conf['d'], 0, 546.],
                                   boom_dimensions={'start_bay': 6})]

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__ because all
        detectors need different parameters. Placing this specific code in it's own
        function makes it easy to override for derived classes.
        '''
        twostrips = DetTwoStrips(conf)
        proj = marxs.analysis.ProjectOntoPlane()
        detfp = FocalPlaneDet()
        return [twostrips, proj, detfp]

    def post_process(self):
        self.KeepPos = KeepCol('pos')
        return [self.KeepPos]

    def __init__(self, conf=defaultconf, channels=['1', '2', '1m', '2m'],
                 **kwargs):
        list_of_classes = [self.aper_class, self.spo_class,
                           self.gratings_class, self.filter_and_qe_class]
        elem = []
        for c in list_of_classes:
            if c is not None:
                elem.append(c(conf, channels))
        elem.extend(self.add_boom(conf))
        elem.extend(self.add_detectors(conf))

        elem.append(tagversion)
        super(Arcus, self).__init__(elements=elem,
                                    postprocess_steps=self.post_process(),
                                    **kwargs)


class ArcusForPlot(Arcus):

    def add_boom(self, conf):
        return [boom.FourSidedBoom(orientation=xyz2zxy[:3, :3],
                                   position=[conf['d'], 0, 546.])]

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__ because all
        detectors need different parameters. Placing this specific code in it's own
        function makes it easy to override for derived classes.
        '''
        return [DetTwoStrips(conf)]


class ArcusForSIXTE(Arcus):
    filter_and_qe_class = None

    def add_detectors(self, conf):
        return [FocalPlaneDet()]
