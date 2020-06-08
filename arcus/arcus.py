from copy import deepcopy
import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d
import transforms3d

import marxs
from marxs.simulator import Sequence, KeepCol
from marxs.optics import (GlobalEnergyFilter,
                          FlatDetector,
                          CircularDetector)
from marxs.math.geometry import Cylinder
from marxs import optics
from marxs.design.rowland import RowlandCircleArray
import marxs.analysis
from marxs.design import tolerancing as tol

from .ralfgrating import (InterpolateRalfTable, RalfQualityFactor,
                          catsupportbars, catsupport,
                          CATfromMechanical, CATWindow)
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
defaultconf = make_rowland_from_d_BF_R_f(600., 5915.51307, 12000. - 123.569239)
defaultconf['blazeang'] = 1.8
defaultconf['n_CCDs'] = 16
defaultconf['phi_det_start'] = 0.037


def reformat_randall_errorbudget(budget, globalfac=0.8):
    '''Reformat the numbers from LSF-CAT-Alignment-v3.xls

    Randall gives 3 sigma errors, while I need 1 sigma a input here.
    Also, units need to be converted: mu -> mm, arcsec -> rad
    Last, global misalignment (that's not random) must be
    scaled in some way. Here, I use 0.8 sigma, which is the
    mean absolute deviation for a Gaussian.

    Parameters
    ----------
    budget : list
        See reference implementation for list format
    globalfac : ``None`` or float
        Factor to apply for global tolerances. A "global" tolerance is drawn
        only once per simulation. In contrast, for "individual" tolerances
        many draws are done and thus the resulting layout actually
        represents a distribution. For a "global" tolerance, the result hinges
        essentially on a single random draw. If this is set to ``None``,
        misalignments are drawn statistically. Instead, the toleracnes can be
        scaled determinisitically, e.g. by "0.8 sigma" (the mean absolute
        deviation for a Gaussian distribution).
    '''
    for row in budget:
        tol = np.array(row[2], dtype=float)
        tol[:3] = tol[:3] / 1000.  # mu to mm
        tol[3:] = np.deg2rad(tol[3:] / 3600.)  # arcsec to rad
        tol = tol / 3   # Randall gives 3 sigma values
        if row[1] == 'global':
            if globalfac is not None:
                tol *= globalfac
            else:
                tol *= np.random.randn(len(tol))

        row[3] = tol


id_num_offset = {'1': 0,
                 '2': 1000,
                 '1m': 10000,
                 '2m': 11000}


# Set a little above entrance pos (the mirror) for display purposes.
# Thus, needs to be geometrically bigger for off-axis sources.
# with fac > 0.5
fac = 1.
spopos = np.array(spo.spo_pos4d)
rmid = 0.5 * (spopos[:, 1, 3].max() + spopos[:, 1, 3].min())
delta_r = spo.spogeom['outer_radius'] - spo.spogeom['inner_radius']
rdim = spopos[:, 1, 3].max() - rmid + fac * delta_r.max()
aperzoom = [1, spopos[:, 0, 3].max() + fac * spo.spogeom['azwidth'].max(),
            rdim
            ]


class Aperture(optics.MultiAperture):
    def __init__(self, conf, channels=['1', '2', '1m', '2m'], **kwargs):
        apers = []
        for chan in channels:
            pos = conf['pos_opt_ax'][chan][:3].copy()
            pos[2] += 12200
            if '1' in chan:
                pos[1] += rmid
            elif '2' in chan:
                pos[1] -= rmid
            else:
                raise ValueError('No rules for channel {}'.format(chan))

            rect = optics.RectangleAperture(position=pos,
                                            zoom=aperzoom,
                                            orientation=xyz2zxy[:3, :3])
            rect.display['outer_factor'] = 2
            apers.append(rect)

        super(Aperture, self).__init__(elements=apers, **kwargs)


def spomounting(photons):
    '''Remove photons that do not go through an SPO but hit the
    frame part of the petal.'''
    photons['probability'][photons['xou'] < 0] = 0.
    return photons


class SimpleSPOs(Sequence):

    def __init__(self, conf, channels=['1', '2', '1m', '2m'],
                 **kwargs):
        rot180 = transforms3d.euler.euler2mat(np.pi, 0, 0, 'szyx')
        # Make lens a little larger than aperture, otherwise an non on-axis ray
        # (from pointing jitter or an off-axis source) might miss the mirror.
        mirror = []
        for chan in channels:
            entrancepos = conf['pos_opt_ax'][chan][:3].copy()
            entrancepos[2] += 12000
            rot = np.eye(3) if '1' in chan else rot180
            mirror.append(spo.SPOChannelMirror(position=entrancepos,
                                               orientation=rot,
                                               id_num_offset=id_num_offset[chan]))
            mirror.append(spo.ScatterPerChannel(position=entrancepos,
                                                min_id=id_num_offset[chan],
                                                inplanescatter=spo.inplanescatter,
                                                perpplanescatter=spo.perpplanescatter,
                                                orientation=xyz2zxy[:3, :3]))
        mirror.append(spomounting)
        mirror.append(spo.geometricthroughput)
        super(SimpleSPOs, self).__init__(elements=mirror, **kwargs)


class CATGratings(Sequence):
    order_selector_class = InterpolateRalfTable
    gratquality_class = RalfQualityFactor

    def __init__(self, conf, channels=['1', '2', '1m', '2m'], **kwargs):

        elements = []

        self.gratquality = self.gratquality_class()
        for chan in channels:
            elements.append(CATfromMechanical(pos4d=conf['shift_optical_axis_' + chan],
                                              channel=chan, conf=conf,
                                              id_num_offset=id_num_offset[chan],
                                              ))
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
    # orientation flips around CCDs so that det_x increases
    # with increasing x coordinate
    elem_args = {'pixsize': 0.024, 'zoom': [1, 24.576, 12.288],
                 'orientation': np.array([[-1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, +1]])}
    d_element = elem_args['zoom'][1] * 2 + 0.824 * 2 + 0.5
    theta = [np.pi - 0.5, np.pi + 0.5]

    def __init__(self, conf, **kwargs):
        super(DetMany, self).__init__(rowland=conf['rowland_detector'],
                                      elem_class=self.elem_class,
                                      elem_args=self.elem_args,
                                      d_element=self.d_element,
                                      theta=self.theta,
                                      # convention is to start counting at 1
                                      id_num_offset=1,
                                      id_col='CCD')


class Det16(DetMany):
    '''Place only hand-selected 16 CCDs'''
    def __init__(self, conf, theta=[3.1255, 3.1853, 3.2416, 3.301], **kwargs):
        self.theta = theta
        super(Det16, self).__init__(conf, **kwargs)
        assert len(self.elements) == 16
        # but make real detectors orange
        disp = deepcopy(self.elements[0].display)
        disp['color'] = 'orange'
        for e in self.elements:
            e.display = disp


class DetCamera(DetMany):
    '''CCD detectors in camera layout

    8 CCDs in each strip, with gaps between them in 3-2-3 groups
    to match the shadows from the filter support.
    One strip is offset by a few mm so that the chip gaps fall
    into slightly different positions.
    '''
    d_fsupport = 4.
    '''Distance between CCDs under a filter support bar in mm'''

    d_ccd = 2.15
    '''Minimal distance between CCDs in mm'''

    offset = 5.
    '''offset of one strip vs the other to avoid matching chip gaps in mm'''

    def __init__(self, conf, **kwargs):
        r = conf['rowland_detector'].r
        phi_m = np.arcsin(conf['d'] / r) + np.pi
        ccd = self.elem_args['zoom'][1]
        p0 = conf['phi_det_start']
        gaps = np.array([0, self.d_ccd, self.d_fsupport, self.d_ccd,
                         self.d_ccd, self.d_fsupport,
                         self.d_ccd, self.d_ccd])
        theta1 = (p0 + ccd / r) + np.arange(8) * 2 * ccd / r + gaps.cumsum() / r
        self.theta = np.hstack([phi_m - theta1,
                                phi_m + self.offset / r + theta1])
        # Sort so that CCD number increases from -x to +x
        self.theta.sort()
        self.theta = self.theta[::-1]

        super(DetCamera, self).__init__(conf, **kwargs)
        assert len(self.elements) == conf['n_CCDs']
        # but make real detectors orange
        disp = deepcopy(self.elements[0].display)
        disp['color'] = 'orange'
        for e in self.elements:
            e.display = disp

    def distribute_elements_on_arc(self):
        '''Distributes elements as evenly as possible along an arc segment.

        Returns
        -------
        theta : np.ndarray
            Theta coordinates of the element *center* positions.
        '''
        return self.theta


class FocalPlaneDet(marxs.optics.FlatDetector):
    loc_coos_name = ['detfp_x', 'detfp_y']
    detpix_name = ['detfppix_x', 'detfppix_y']

    def __init__(self, **kwargs):
        if ('zoom' not in kwargs) and ('pos4d' not in kwargs):
            kwargs['zoom'] = [.2, 10000, 10000]
        if ('orientation' not in kwargs) and ('pos4d' not in kwargs):
            kwargs['orientation'] = xyz2zxy[:3, :3]
        super(FocalPlaneDet, self).__init__(**kwargs)


class PerfectArcus(Sequence):
    '''Default Definition of Arcus without any misalignments'''
    aper_class = Aperture
    spo_class = SimpleSPOs
    gratings_class = CATGratings
    filter_and_qe_class = FiltersAndQE
    '''Set any of these classes to None to not have them included.
    (e.g. SIXTE does filters and QE itself).
    '''

    list_of_classes = ['aper_class', 'spo_class', 'gratings_class',
                       'filter_and_qe_class']

    def add_boom(self, conf):
        '''Add four sided boom. Only the top two bays contribute any
        absorption, so we can save time by not modelling the remaining bays.'''
        return [boom.FourSidedBoom(orientation=xyz2zxy[:3, :3],
                                   position=[0, 0, 546.],
                                   boom_dimensions={'start_bay': 6})]

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__ because all
        detectors need different parameters. Placing this specific code in it's own
        function makes it easy to override for derived classes.
        '''
        circdet = CircularDetector(geometry=Cylinder.from_rowland(conf['rowland_detector'],
                                                                  width=20, rotation=np.pi))
        circdet.display['opacity'] = 0.1
        circdet.detpix_name = ['circpix_x', 'circpix_y']
        circdet.loc_coos_name = ['circ_phi', 'circ_y']
        circproj = marxs.analysis.ProjectOntoPlane(orientation=xyz2zxy[:3, :3])
        circproj.loc_coos_name = ['projcirc_x', 'projcirc_y']
        reset = marxs.simulator.simulator.Propagator(distance=-100.)
        twostrips = DetCamera(conf)
        proj = marxs.analysis.ProjectOntoPlane(orientation=xyz2zxy[:3, :3])
        detfp = FocalPlaneDet()
        return [circdet, circproj, reset, twostrips, proj, detfp]

    def post_process(self):
        self.KeepPos = KeepCol('pos')
        return [self.KeepPos]

    def __init__(self, conf=defaultconf, channels=['1', '2', '1m', '2m'],
                 **kwargs):
        elem = []
        for c in self.list_of_classes:
            cl = getattr(self, c)
            if cl is not None:
                elem.append(cl(conf, channels))
        elem.extend(self.add_boom(conf))
        elem.extend(self.add_detectors(conf))
        elem.append(tagversion)
        super(PerfectArcus, self).__init__(elements=elem,
                                           postprocess_steps=self.post_process(),
                                           **kwargs)


class Arcus(PerfectArcus):
    def __init__(self, conf=defaultconf, channels=['1', '2', '1m', '2m'],
                 **kwargs):
        super(Arcus, self).__init__(conf=conf, channels=channels, **kwargs)
        for row in conf['alignmentbudget']:
            elem = self.elements_of_class(row[0])
            if row[1] == 'global':
                tol.moveglobal(elem, *row[3])
            elif row[1] == 'individual':
                tol.wiggle(elem, *row[3])
            else:
                raise NotImplementedError('Alignment error {} not implemented'.format(row[1]))


class ArcusForPlot(PerfectArcus):
    '''Arcus with setting that are good for 3D plots

    In particular, there is a full boom and no large catch-all focal plane.
    '''
    def add_boom(self, conf):
        return [boom.FourSidedBoom(orientation=xyz2zxy[:3, :3],
                                   position=[0, 0, 546.])]

    def add_detectors(self, conf):
        '''Add detectors to the element list

        This is a separate function that is called from __init__ because all
        detectors need different parameters. Placing this specific code in it's own
        function makes it easy to override for derived classes.
        '''
        return [DetCamera(conf)]


class ArcusForSIXTE(Arcus):
    filter_and_qe_class = None

    def add_detectors(self, conf):
        return [FocalPlaneDet()]


align_requirement_smith = [
    [spo.SPOChannelMirror, 'individual', [12.5, 100, 50, 300, 300, 10],
     None, 'individual SPO in petal'],
    [spo.SPOChannelMirror, 'global', [0., 0, 0, 0, 0, 0],
     None, 'SPO petal to front assembly'],
    [CATfromMechanical, 'global', [1000, 1000, 1000, 300, 300, 600],
     None, 'CAT petal to SPO petal'],
    [CATfromMechanical, 'individual', [1000, 1000, 200, 300., 180, 300],
     None, 'CAT windows to CAT petal'],
    [CATWindow, 'individual', [1000, 1000, 200, 300, 180, 300],
     None, 'individual CAT to window'],
    [DetCamera, 'global', [5000, 2000, 1000, 180, 180, 180],
     None, 'Camera to front assembly']]
'''This is taken from LSF-CAT-Alignment-v3.xls from R. Smith'''

align_requirement = deepcopy(align_requirement_smith)
reformat_randall_errorbudget(align_requirement)

defaultconf['alignmentbudget'] = align_requirement
