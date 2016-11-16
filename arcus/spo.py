import numpy as np
from transforms3d.euler import euler2mat
from astropy.table import Table

from marxs.optics.aperture import RectangleAperture, MultiAperture
from marxs.optics.base import FlatOpticalElement
from marxs.optics import RadialMirrorScatter, PerfectLens
from marxs.base import _parse_position_keywords


class SPOModule(RectangleAperture):
    _plot_mayavi = FlatOpticalElement._plot_mayavi
    _plot_threejs = FlatOpticalElement._plot_threejs

    display = {'color': (1.0, 1.0, 0.0)}

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


class SPOChannel(MultiAperture):
    def __init__(self, **kwargs):
        self.pos4d = _parse_position_keywords(kwargs)
        spogeom = Table.read(spo_geometry, format='ascii')
        elements = []
        for i, row in enumerate(spo_angles):
            for ang in row:
                zoom = [spogeom[i]['height'] / 2.,
                        spogeom[i]['width'] / 2.,
                        spogeom[i]['depth'] / 2.]
                z = spogeom[i]['r_mid'] * np.cos(np.deg2rad(ang))
                y = spogeom[i]['r_mid'] * np.sin(np.deg2rad(ang))
                pos = [focallength - spogeom[i]['d_from_12m'], y, z]
                orient = euler2mat(-np.deg2rad(ang), 0., 0.)
                spo = SPOModule(position=pos, zoom=zoom, orientation=orient)
                elements.append(spo)
        kwargs['elements'] = elements
        super(SPOChannel, self).__init__(**kwargs)
        for e in self.elements:
            e.pos4d = np.dot(self.pos4d, e.pos4d)

        shift_by_f = np.eye(4)
        shift_by_f[0, 3] = focallength
        mirrorcenter = np.dot(self.pos4d, shift_by_f)
        self.mirror = PerfectLens(pos4d=mirrorcenter, focallength=focallength)
        self.scatter = RadialMirrorScatter(pos4d=mirrorcenter,
                                           inplanescatter=inplanescatter,
                                           perpplanescatter=perpplanescatter)

    def __call__(self, photons):
        photons = super(SPOChannel, self)(photons)
        photons = self.mirror(photons)
        photons = self.scatter(photons)
        return photons
