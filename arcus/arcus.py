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
from marxs.design.rowland import (RowlandTorus,
                                  design_tilted_torus,
                                  RowlandCircleArray)
import marxs.analysis

from ralfgrating import (InterpolateRalfTable, RalfQualityFactor,
                         catsupportbars, catsupport,
                         RectangularGrid)
import spo
from .load_csv import load_number, load_table
from .utils import tagversion

__all__ = ['xyz2zxy',
           'jitter_sigma',
           'Arcus', 'ArcusForPlot', 'ArcusForSIXTE']

jitter_sigma = load_number('other', 'pointingjitter',
                           'FWHM') / 2.3545


geometry = {'blazeang': 1.91, # Blaze angle in degrees
            'offset_spectra': 5., # Offset of each of the two spectra from CCD center
            'alpha': np.deg2rad(2.2 * 1.91),
            'beta': np.deg2rad(4.4 * 1.91),
            'max_z_torus': 11.9e3}

xyz2zxy = np.array([[ 0.,  1.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.],
                    [ 1.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  1.]])

def derive_rowland_and_shiftmatrix(geometry):
    R, r, pos4d = design_tilted_torus(geometry['max_z_torus'],
                                      geometry['alpha'],
                                      geometry['beta'])
    out = {'rowland_central': RowlandTorus(R, r, pos4d=pos4d)}
    out['rowland_central'].pos4d = np.dot(xyz2zxy, out['rowland_central'].pos4d)

    # Now offset that Rowland torus in a z axis by a few mm.
    # Shift is measured from a focal point that hits the center of the CCD strip.
    out['shift_optical_axis_1'] = np.eye(4)
    out['shift_optical_axis_1'][1, 3] = - geometry['offset_spectra']

    out['rowland'] = RowlandTorus(R, r, pos4d=pos4d)
    out['rowland'].pos4d = np.dot(xyz2zxy, out['rowland'].pos4d)
    out['rowland'].pos4d = np.dot(out['shift_optical_axis_1'], out['rowland'].pos4d)


    Rm, rm, pos4dm = design_tilted_torus(geometry['max_z_torus'],
                                         - geometry['alpha'],
                                         - geometry['beta'])
    out['rowlandm'] = RowlandTorus(Rm, rm, pos4d=pos4dm)
    out['rowlandm'].pos4d = np.dot(xyz2zxy, out['rowlandm'].pos4d)

    out['d'] = r * np.sin(geometry['alpha'])
    # Relative to origin in the center of the CCD strip
    out['shift_optical_axis_2'] = np.eye(4)
    out['shift_optical_axis_2'][0, 3] = 2. * out['d']
    out['shift_optical_axis_2'][1, 3] = + geometry['offset_spectra']

    # Optical axis 2 relative to optical axis 1
    out['shift_optical_axis_12'] = np.dot(np.linalg.inv(out['shift_optical_axis_1']),
                                          out['shift_optical_axis_2'])

    out['rowlandm'].pos4d = np.dot(out['shift_optical_axis_2'],
                                   out['rowlandm'].pos4d)

    return out


defaultconf = deepcopy(geometry)
defaultconf.update(derive_rowland_and_shiftmatrix(defaultconf))


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
        if '1' in channels:
            mirror.append(spo.SPOChannelMirror(position=entrancepos,
                                               orientation=xyz2zxy[:3, :3],
                                               id_num_offset=0))
        if '2' in channels:
            mirror.append(spo.SPOChannelMirror(position=entrancepos,
                                               orientation=np.dot(rot180, xyz2zxy[:3, :3]),
                                               id_num_offset=1000))
        if '1m' in channels:
            mirror.append(spo.SPOChannelMirror(position=entranceposm,
                                               orientation=xyz2zxy[:3, :3],
                                               id_num_offset=10000))
        if '2m' in channels:
            mirror.append(spo.SPOChannelMirror(position=entranceposm,
                                               orientation=np.dot(rot180, xyz2zxy[:3, :3]),
                                               id_num_offset=11000))
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
        mirror.append(spomounting)
        super(SimpleSPOs, self).__init__(elements=mirror, **kwargs)


class CATGratings(Sequence):
    order_selector_class = InterpolateRalfTable
    gratquality_class = RalfQualityFactor

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
                       'elem_args': {'d': 2e-4, 'zoom': [1., 15., 15.],
                                     'orientation': blazemat,
                                     'order_selector': self.order_selector},
                       'normal_spec': np.array([0, -conf['offset_spectra'], 0., 1.]),
                       'parallel_spec': np.array([1., 0., 0., 0.])
        }
        y_offset = conf['offset_spectra']
        d = conf['d']
        if '1' in channels:
            elements.append(RectangularGrid(y_range=[300 - y_offset, 900 - y_offset],
                                            x_range=[-180, 180], **gratinggrid))
        if '2' in channels:
            elements.append(RectangularGrid(y_range=[-900 - y_offset, -300 - y_offset],
                                            x_range=[-180, 180],
                                            id_num_offset=1000, **gratinggrid))
        if ('1m' in channels) or ('2m' in channels):
            gratinggrid['rowland'] = conf['rowlandm']
            gratinggrid['elem_args']['orientation'] = blazematm
            gratinggrid['normal_spec'] = np.array([2 * conf['d'], y_offset, 0., 1.])
        if '1m' in channels:
            elements.append(RectangularGrid(y_range=[300 + y_offset, 900 + y_offset],
                                            x_range=[-180 + 2 * d, 180 + 2 * d],
                                            id_num_offset=10000, **gratinggrid))
        if '2m' in channels:
            elements.append(RectangularGrid(y_range=[-900 + y_offset, -300 + y_offset],
                                            x_range=[-180 + 2* d, 180 + 2 * d],
                                            id_num_offset=11000, **gratinggrid))
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
    theta = [np.pi - 0.2, np.pi + 0.5]

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

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__ because all
        detectors need different parameters. Placing this specific code in it's own
        function makes it easy to override for derived classes.
        '''
        det16 = Det16(conf)
        proj = marxs.analysis.ProjectOntoPlane()
        detfp = FocalPlaneDet()
        return [det16, proj, detfp]

    def post_process(self):
        self.KeepPos = KeepCol('pos')
        return [self.KeepPos]

    def __init__(self, conf=defaultconf, channels=['1', '2', '1m', '2m'], **kwargs):
        list_of_classes = [self.aper_class, self.spo_class,
                           self.gratings_class, self.filter_and_qe_class]
        elem = []
        for c in list_of_classes:
            elem.append(c(conf, channels))
        elem.extend(self.add_detectors(conf))

        elem.append(tagversion)
        super(Arcus, self).__init__(elements=elem,
                                    postprocess_steps=self.post_process(),
                                    **kwargs)


class ArcusForPlot(Arcus):
    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__ because all
        detectors need different parameters. Placing this specific code in it's own
        function makes it easy to override for derived classes.
        '''
        return [Det16(conf)]


class ArcusForSIXTE(Arcus):
    def __init__(self, conf=defaultconf, channels=['1', '2', '1m', '2m'], **kwargs):
        elem = [c(conf, channels) for c in [self.aper_class, self.spo_class,
                                            self.gratings_class, FocalPlaneDet()]]
        elem.append(tagversion)
        super(ArcusForSIXTE, self).__init__(elements=elem,
                                    postprocesss_steps=self.post_process()
                                    **kwargs)
