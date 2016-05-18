import numpy as np
import xlrd
import astropy.units as u
from scipy.interpolate import interp1d, RectBivariateSpline


import marxs
import marxs.optics
from marxs.simulator import Sequence
from marxs.source import PointSource, FixedPointing
from marxs.optics import EnergyFilter, FlatDetector, CATGrating
from marxs.design.rowland import RowlandTorus, design_tilted_torus, GratingArrayStructure

from read_grating_data import InterpolateRalfTable

# Reading in data for grating reflectivity, filtercurves etc.
arcusefficiencytable = xlrd.open_workbook('ArcusEffectiveArea-v3.xls')
mastersheet = arcusefficiencytable.sheet_by_name('Master')
energy = np.array(mastersheet.col_values(0, start_rowx=6)) / 1000.  # ev to keV
spogeometricthroughput = np.array(mastersheet.col_values(3, start_rowx=6))
doublereflectivity = np.array(mastersheet.col_values(4, start_rowx=6))
sifiltercurve = np.array(mastersheet.col_values(20, start_rowx=6))
uvblocking = np.array(mastersheet.col_values(21, start_rowx=6))
opticalblocking = np.array(mastersheet.col_values(22, start_rowx=6))
ccdcontam = np.array(mastersheet.col_values(23, start_rowx=6))
qebiccd = np.array(mastersheet.col_values(24, start_rowx=6))

# Define filters as optical elements
mirrorefficiency = EnergyFilter(filterfunc=interp1d(energy, spogeometricthroughput * doublereflectivity))

star = PointSource(coords=(23., 45.), flux=5.)

pointing = FixedPointing(coords=(23., 45.), roll=0.)
entrancepos = np.array([12000., 0., 0.])

aper = marxs.optics.CircleAperture(position=[12000, 0, 0], zoom=1000)
lens = marxs.optics.PerfectLens(focallength=12000., position=entrancepos)
rms = marxs.optics.RadialMirrorScatter(inplanescatter=(24 * u.arcsec).to(u.radian).value,
                                       perpplanescatter=(1.05 * u.arcsec).to(u.radian).value,
                                       position=entrancepos)
mirror = Sequence(sequence=[lens, rms, mirrorefficiency])

# CAT grating
order_selector = InterpolateRalfTable('Si_4um_deep_30pct_dc.xlsx')

# Define L1, L2 blockage as simple filters due to geometric area
# L1 support: blocks 18 %
# L2 support: blocks 19 %
catsupport = EnergyFilter(filterfunc=lambda e: 0.81 * 0.82)

R, r, pos4d = rowland_pars = design_tilted_torus(12e3, 2 * np.deg2rad(1.91),
                                                 5 * np.deg2rad(1.91))
rowland = RowlandTorus(R, r, pos4d=pos4d)
gas = GratingArrayStructure(rowland, d_facet=22., x_range=[1e4, 1.2e4],
                            radius=[300, 400], phi=[0, 2. * np.pi],
                            elem_class=CATGrating,
                            elem_args={'d': 2e-4, 'zoom': 10.,
                                       'order_selector': order_selector},
                            )


det = marxs.optics.FlatStack(zoom=1000,
                             sequence=[EnergyFilter, FlatDetector],
                             keywords=[{'filterfunc': interp1d(energy, sifiltercurve * uvblocking * opticalblocking * ccdcontam * qebiccd)}, {}])

arcus = Sequence(sequence=[pointing, aper, mirror, gas, catsupport, det])

photons = star.generate_photons(exposuretime=5000)
p = arcus(photons)
