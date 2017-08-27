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
                                  RectangularGrid,
                                  RowlandCircleArray)
import marxs.analysis

from ralfgrating import (InterpolateRalfTable, RalfQualityFactor,
                         catsupportbars, catsupport)
from spo import SPOChannelMirror, spogeometricthroughput, doublereflectivity
from .load_csv import load_number, load_table
from .utils import tagversion

jitter_sigma = load_number('other', 'pointingjitter',
                           'FWHM') / 2.3545

# Blaze angle in degrees
blazeang = 1.91

# We want two different spectral traces in the same array of chips.
# Offset them a little so that the spectra can be separated easily in data analysis.
# CCDs are ~25 mm wide.
z_offset_spectra = 5.

alpha = np.deg2rad(2.2 * blazeang)
beta = np.deg2rad(4.4 * blazeang)
max_x_torus = 11.9e3

R, r, pos4d = design_tilted_torus(11.9e3, alpha, beta)
rowland_central = RowlandTorus(R, r, pos4d=pos4d)

# Now offset that Rowland torus in a z axis by a few mm.
# Shift is measured from a focal point that hits the center of the CCD strip.
shift_optical_axis_1 = np.eye(4)
shift_optical_axis_1[2, 3] = - z_offset_spectra

rowland = RowlandTorus(R, r, pos4d=pos4d)
rowland.pos4d = np.dot(shift_optical_axis_1, rowland.pos4d)


Rm, rm, pos4dm = design_tilted_torus(max_x_torus, - alpha, -beta)
rowlandm = RowlandTorus(Rm, rm, pos4d=pos4dm)
d = r * np.sin(alpha)
# Relative to z=0 in the center of the CCD strip
shift_optical_axis_2 = np.eye(4)
shift_optical_axis_2[1, 3] = 2. * d
shift_optical_axis_2[2, 3] = + z_offset_spectra

# Relative to optical axis 1
shift_optical_axis_12 = np.eye(4)
shift_optical_axis_12[1, 3] = 2. * d
shift_optical_axis_12[2, 3] = 2. * z_offset_spectra

rowlandm.pos4d = np.dot(shift_optical_axis_2, rowlandm.pos4d)


entrancepos = np.array([12000., 0., -z_offset_spectra])

# Set a little above entrance pos (the mirror) for display purposes.
# Thus, needs to be geometrically bigger for off-axis sources.

# aper = optics.CircleAperture(position=[12200, 0, 0], zoom=300,
#                       phi=[-0.3 + np.pi / 2, .3 + np.pi / 2])\

aper_rect1 = optics.RectangleAperture(position=[12200, 0, 550], zoom=[1, 180, 250])
aper_rect2 = optics.RectangleAperture(position=[12200, 0, -550], zoom=[1, 180, 250])

aper_rect1m = optics.RectangleAperture(pos4d=np.dot(shift_optical_axis_12, aper_rect1.pos4d))
aper_rect2m = optics.RectangleAperture(pos4d=np.dot(shift_optical_axis_12, aper_rect2.pos4d))

aper = optics.MultiAperture(elements=[aper_rect1, aper_rect2])
aperm = optics.MultiAperture(elements=[aper_rect1m, aper_rect2m])
aper4 = optics.MultiAperture(elements=[aper_rect1, aper_rect2,
                                       aper_rect1m, aper_rect2m])

# Make lens a little larger than aperture, otherwise an non on-axis ray
# (from pointing jitter or an off-axis source) might miss the mirror.
lens1 = SPOChannelMirror(position=entrancepos,
                         id_num_offset=0)
lens2 = SPOChannelMirror(position=entrancepos,
                         orientation=transforms3d.euler.euler2mat(np.pi, 0,0,'sxyz'),
                         id_num_offset=1000)
lens1m = SPOChannelMirror(pos4d=np.dot(shift_optical_axis_12, lens1.pos4d),
                          id_num_offset=10000)
lens2m = SPOChannelMirror(pos4d=np.dot(shift_optical_axis_12, lens2.pos4d),
                          id_num_offset=11000)
# Scatter as FWHM ~8 arcsec. Divide by 2.3545 to get Gaussian sigma.
rms = RadialMirrorScatter(inplanescatter=10. / 2.3545 / 3600 / 180. * np.pi,
                          perpplanescatter=1.5 / 2.345 / 3600. / 180. * np.pi,
                          position=entrancepos, zoom=[1, 200, 820])

rmsm = RadialMirrorScatter(inplanescatter=10. / 2.3545 / 3600 / 180. * np.pi,
                           perpplanescatter=1.5 / 2.345 / 3600. / 180. * np.pi,
                           pos4d=np.dot(shift_optical_axis_12, rms.pos4d))


def spomounting(photons):
    '''Remove photons that do not go through an SPO but hit the
    frame part of the petal.'''
    photons['probability'][photons['spo'] < 0] = 0.
    return photons


mirror = Sequence(elements=[lens1, lens2, rms,
                            spogeometricthroughput, doublereflectivity,
                            spomounting])
mirrorm = Sequence(elements=[lens1m, lens2m, rmsm,
                             spogeometricthroughput, doublereflectivity,
                             spomounting])
mirror4 = Sequence(elements=[lens1, lens2, lens1m, lens2m,
                             rms, rmsm,
                             spogeometricthroughput, doublereflectivity,
                             spomounting])

# CAT grating
order_selector = InterpolateRalfTable()
gratquality = RalfQualityFactor()

blazemat = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]),
                                             np.deg2rad(-blazeang))
blazematm = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]),
                                              np.deg2rad(blazeang))

gratinggrid = {'rowland': rowland, 'd_element': 32., 'x_range': [1e4, 1.4e4],
               'elem_class': CATGrating,
               'elem_args': {'d': 2e-4, 'zoom': [1., 15., 15.],
                             'orientation': blazemat,
                             'order_selector': order_selector},
               'normal_spec': np.array([0, 0., -z_offset_spectra, 1.])
              }
gas_1 = RectangularGrid(z_range=[300 - z_offset_spectra, 880 - z_offset_spectra],
                        y_range=[-180, 180], **gratinggrid)
gas_2 = RectangularGrid(z_range=[-880 + z_offset_spectra, -300 - z_offset_spectra],
                        y_range=[-180, 180],
                        id_num_offset=1000, **gratinggrid)
gas = Sequence(elements=[gas_1, gas_2, catsupport, catsupportbars, gratquality])

gratinggrid['rowland'] = rowlandm
gratinggrid['elem_args']['orientation'] = blazematm
gratinggrid['normal_spec'] = np.array([0, 2 * d, z_offset_spectra, 1.])
gas_1m = RectangularGrid(z_range=[300 + z_offset_spectra, 880 + z_offset_spectra],
                         y_range=[-180 + 2 * d, 180 + 2 * d],
                         id_num_offset=10000, **gratinggrid)
gas_2m = RectangularGrid(z_range=[-880 + z_offset_spectra, -300 + z_offset_spectra],
                         y_range=[-180 + 2* d, 180 + 2 * d],
                         id_num_offset=11000, **gratinggrid)
gasm = Sequence(elements=[gas_1m, gas_2m,
                          catsupport, catsupportbars, gratquality])
gas4 = Sequence(elements=[gas_1, gas_2, gas_1m, gas_2m,
                          catsupport, catsupportbars, gratquality])


def get_filter(dir, name):
    tab = load_table(dir, name)
    en = tab['energy'].to(u.keV, equivalencies=u.spectral())
    return GlobalEnergyFilter(filterfunc=interp1d(en, tab[tab.colnames[1]]),
                              name=name)

filtersandqe = Sequence(elements=[get_filter(*n) for n in [('filters', 'sifilter'),
                                                           ('filters', 'opticalblocking'),
                                                           ('filters', 'uvblocking'),
                                                           ('detectors', 'contam'),
                                                           ('detectors', 'qe')]])

detccdargs = {'pixsize': 0.024,'zoom': [1, 24.576, 12.288]}

# 500 mu gap between detectors
# Place only hand-selected 16 CCDs
det_16 = RowlandCircleArray(rowland=rowland_central,
                            elem_class=FlatDetector,
                            elem_args=detccdargs,
                            d_element=49.652, theta=[3.1255, 3.1853, 3.2416, 3.301])
assert len(det_16.elements) == 16

# Put plenty of CCDs in the focal plane
det = RowlandCircleArray(rowland=rowland_central,
                         elem_class=FlatDetector,
                         elem_args=detccdargs,
                         d_element=49.652, theta=[np.pi - 0.2, np.pi + 0.5])

# This is just one way to establish a global coordinate system for
# detection on detectors that follow a curved surface.
# Project (not propagate) down to the focal plane.
projectfp = marxs.analysis.ProjectOntoPlane()

# Place an additional detector on the Rowland circle.
detcirc = marxs.optics.CircularDetector.from_rowland(rowland_central, width=20)
detcirc.loc_coos_name = ['detccent_phi', 'detccent_y']
detcirc.detpix_name = ['detccentpix_x', 'detccentpix_y']
detcirc.display['opacity'] = 0.1  # for all elements of this class

# Place an additional detector on the Rowland circle.
detcirc2 = marxs.optics.CircularDetector.from_rowland(rowlandm, width=20)
detcirc2.loc_coos_name = ['detc2_phi', 'detc2_y']
detcirc2.detpix_name = ['detc2pix_x', 'detc2pix_y']

# Place an additional detector on the Rowland circle.
detcirc1 = marxs.optics.CircularDetector.from_rowland(rowland, width=20)
detcirc1.loc_coos_name = ['detc1_phi', 'detc1_y']
detcirc1.detpix_name = ['detc1_x', 'detc1_y']


# Place an additional detector in the focal plane for comparison
# Detectors are transparent to allow this stuff
detfp = marxs.optics.FlatDetector(zoom=[.2, 10000, 10000])
detfp.loc_coos_name = ['detfp_x', 'detfp_y']
detfp.detpix_name = ['detfppix_x', 'detfppix_y']
detfp.display = deepcopy(detfp.display)
detfp.display['opacity'] = 0.1

# but make real detectors orange
det.elements[0].display['color'] = 'orange'

### Put together ARCUS in different configurations ###
arcus = Sequence(elements=[aper, mirror, gas, filtersandqe, det_16,
                           projectfp, tagversion])
arcusm = Sequence(elements=[aperm, mirrorm, gasm, filtersandqe, det_16,
                            projectfp, tagversion])
keeppos4 = KeepCol('pos')
arcus4 = Sequence(elements=[aper4, mirror4, gas4, filtersandqe, det_16,
                            projectfp, tagversion],
                  postprocess_steps=[keeppos4])
arcus_for_plot = Sequence(elements=[aper, aperm, gas, gasm, det_16,
                                    tagversion])

keeppos = KeepCol('pos')
keepposm = KeepCol('pos')

arcus_extra_det4 = Sequence(elements=[aper4, mirror4, gas4, filtersandqe,
                                      detcirc, detcirc1, detcirc2, det,
                                      projectfp, detfp,
                                      tagversion],
                            postprocess_steps=[keeppos])

arcus_extra_det = Sequence(elements=[aper, mirror, gas, filtersandqe,
                                     detcirc, detcirc1, detcirc2, det,
                                     projectfp, detfp,
                                     tagversion],
                           postprocess_steps=[keeppos])

arcus_extra_det_m = Sequence(elements=[aperm, mirrorm, gasm, filtersandqe,
                                       detcirc, detcirc1, detcirc2, det,
                                       projectfp, detfp,
                                       tagversion],
                             postprocess_steps=[keepposm])

# No detector effects - Joern's simulator handles that itself.
arcus_joern = Sequence(elements=[aper, mirror, gas, detfp, tagversion])
arcus_joernm = Sequence(elements=[aperm, mirrorm, gasm, detfp, tagversion])
