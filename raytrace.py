import xlrd
import astropy.units as u

import marxs
import marxs.optics
from marxs.simulator import Sequence
from marxs.source import PointSource, FixedPointing

# Reading in data for grating reflectivity, filtercurves etc.
arcusefficiencytable = xlrd.open_workbook('ArcusEffectiveArea-v3.xls')
mastersheet = arcusefficiencytable.sheet_by_name('Master')
energy = mastersheet.col_values(0, start_rowx=6)
geometricthroughput = mastersheet.col_values(3, start_rowx=6)
doublereflectivity = mastersheet.col_values(4, start_rowx=6)
sifiltercurve = mastersheet.col_values(19, start_rowx=6)
geometricthroughput = mastersheet.col_values(20, start_rowx=6)
uvblocking = mastersheet.col_values(21, start_rowx=6)
opticalblocking = mastersheet.col_values(22, start_rowx=6)
ccdcontam = mastersheet.col_values(23, start_rowx=6)
qebiccd = mastersheet.col_values(24, start_rowx=6)


star = PointSource(coords=(23., 45.), flux=5.)

pointing = FixedPointing(coords=(23., 45.), roll=0.)
entrancepos = np.array([12000., 0., 0.])

aper = marxs.optics.RectangleAperture(position=[12000, 0, 0], zoom=1000)
mirr = marxs.optics.PerfectLens(focallength=12000., position=entrancepos)
rms = marxs.optics.RadialMirrorScatter(inplanescatter=(24 * u.arcsec).to(u.radian).value,
                                       perpplanescatter=(1.05 * u.arcsec).to(u.radian).value,
                                       position=entrancepos)

det = marxs.optics.FlatDetector(position=[0, 0, 0], zoom=1000)

arcus = Sequence(sequence=[pointing, aper, mirr, rms, det])

photons = star.generate_photons(5000)
p = arcus(photons)



