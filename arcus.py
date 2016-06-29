import os

import numpy as np
import xlrd
from scipy.interpolate import interp1d
import transforms3d

import marxs
from marxs.simulator import Sequence
from marxs.optics import (GlobalEnergyFilter, EnergyFilter,
                          FlatDetector, CATGrating,
                          CircleAperture, PerfectLens, RadialMirrorScatter)
from marxs.design.rowland import (RowlandTorus, design_tilted_torus,
                                  GratingArrayStructure, LinearCCDArray)

from read_grating_data import InterpolateRalfTable

path = os.path.dirname(__file__)

# Reading in data for grating reflectivity, filtercurves etc.
arcusefficiencytable = xlrd.open_workbook(os.path.join(path, '../ArcusEffectiveArea-v3.xls'))
mastersheet = arcusefficiencytable.sheet_by_name('Master')
energy = np.array(mastersheet.col_values(0, start_rowx=6)) / 1000.  # ev to keV
spogeometricthroughput = np.array(mastersheet.col_values(3, start_rowx=6))
doublereflectivity = np.array(mastersheet.col_values(4, start_rowx=6))
sifiltercurve = np.array(mastersheet.col_values(20, start_rowx=6))
uvblocking = np.array(mastersheet.col_values(21, start_rowx=6))
opticalblocking = np.array(mastersheet.col_values(22, start_rowx=6))
ccdcontam = np.array(mastersheet.col_values(23, start_rowx=6))
qebiccd = np.array(mastersheet.col_values(24, start_rowx=6))


mirrorefficiency = GlobalEnergyFilter(filterfunc=interp1d(energy, spogeometricthroughput * doublereflectivity))

entrancepos = np.array([12000., 0., 0.])

# Set a little above entrance pos (the mirror) for display purposes.
# Thus, needs to be geometrically bigger for off-axis sources.

aper = CircleAperture(position=[12200, 0, 0], zoom=300,
                      phi=[-0.3 + np.pi / 2, .3 + np.pi / 2])
lens = PerfectLens(focallength=12000., position=entrancepos)
# Scatter as FWHM ~8 arcsec. Divide by 2.3545 to get Gaussian sigma.
rms = RadialMirrorScatter(inplanescatter=8. / 2.3545 / 3600 / 180. * np.pi,
                          perpplanescatter=1. / 2.345 / 3600. / 180. * np.pi,
                          position=entrancepos)

mirror = Sequence(elements=[lens, rms, mirrorefficiency])

# CAT grating
ralfdata = os.path.join(path, '../Si_4um_deep_30pct_dc.xlsx')
order_selector = InterpolateRalfTable(ralfdata)

# Define L1, L2 blockage as simple filters due to geometric area
# L1 support: blocks 18 %
# L2 support: blocks 19 %
catsupport = GlobalEnergyFilter(filterfunc=lambda e: 0.81 * 0.82)

blazeang = 1.91

R, r, pos4d = design_tilted_torus(12e3, np.deg2rad(blazeang),
                                  2 * np.deg2rad(blazeang))
rowland = RowlandTorus(R, r, pos4d=pos4d)

blazemat = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]), np.deg2rad(-blazeang))
gas = GratingArrayStructure(rowland=rowland, d_element=30.,
                            x_range=[1e4, 1.4e4],
                            radius=[50, 300], phi=[-0.3+np.pi/2, .3+np.pi/2],
                            elem_class=CATGrating,
                            elem_args={'d': 2e-4, 'zoom': [1., 10., 10.], 'orientation': blazemat,
                                       'order_selector': order_selector},
                        )


flatstackargs = {'zoom': [1, 24.576, 12.288],
                 'elements': [EnergyFilter, FlatDetector],
                 'keywords': [{'filterfunc': interp1d(energy, sifiltercurve * uvblocking * opticalblocking * ccdcontam * qebiccd)}, {'pixsize': 0.024}]
                 }
# 500 mu gap between detectors
det = LinearCCDArray(rowland=rowland, elem_class=marxs.optics.FlatStack,
                     elem_args=flatstackargs, d_element=49.652, phi=0,
                     x_range=[0, 200], radius=[-1000, -400])

arcus = Sequence(elements=[aper, mirror, gas, catsupport, det])
