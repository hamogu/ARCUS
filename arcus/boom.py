import numpy as np

from transforms3d.euler import euler2mat
from transforms3d.affines import compose
from marxs.optics import OpticalElement
from marxs.math import pluecker
from marxs.math.utils import h2e, norm_vector
from marxs.simulator import Parallel


class Rod(OpticalElement):
    '''X-axis of the rod is the cylinder axis (x-zoom gives half-length)
    y zoom and z zoom have to be the same (no elliptical cylinders)
    '''
    col_name = 'hitrod'
    display = {'shape': 'cylinder', 'color': 'black'}

    def intersect(self, dir, pos):
        '''Calculate the intersection point between a ray and the element

        Parameters
        ----------
        dir : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the direction of the ray
        pos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of a point on the ray

        Returns
        -------
        intersect :  boolean array of length N
            ``True`` if an intersection point is found.
        interpos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the intersection point. Values are set
            to ``np.nan`` if no intersection point is found.
        interpos_local : `numpy.ndarray` of shape (N, 2)
            y and z coordinates in the coordiante system of the active plane.
        '''
        p_rays = pluecker.dir_point2line(h2e(dir), h2e(pos))
        radius = np.linalg.norm(self.geometry('v_y'))
        height = np.linalg.norm(self.geometry('v_x'))
        intersect = np.zeros(pos.shape[0], dtype=bool)

        # ray passes through cylinder caps?
        for fac in [-1, 1]:
            cap_midpoint = self.geometry('center') + fac * self.geometry('v_x')
            cap_plane = pluecker.point_dir_2plane(cap_midpoint, self.geometry('e_x'))
            interpos = pluecker.intersect_line_plane(p_rays, cap_plane)
            r = np.linalg.norm(h2e(cap_midpoint) - h2e(interpos))
            intersect[r < radius]  = True

        # Ray passes through the side of a cylinder
        # Note that we don't worry about rays parallel to x because those are
        # tested by passing through the caps already
        n = norm_vector(np.cross(self.geometry('e_x'), h2e(dir)))
        d = np.abs(np.dot(n, h2e(self.geometry('center')) - h2e(pos)))
        n2 = norm_vector(np.cross(h2e(dir), n))
        k = np.dot(h2e(pos) - h2e(self.geometry('center')), n2) / np.dot(self.geometry('e_x'), n2)
        intersect[(d < radius) & np.abs(k) < height] = True

        return intersect, None, None

    def process_photons(self, photons, intersect, interpos, intercoos):
        if not self.colname in photons.names:
            photons[self.colname] = False
        photons[self.colname][intersect] = True


# These numbers are here for my notes, not because they can be changed.
# Below, I've multiplied out some of the sin/cos needed to calculate the
# dimensions.
# The boom is unlikely to change in dimension and if it does, all this has to
# be redone anyway.

l_longeron = 1.08 * 1e3
l_batten = 1.6 * 1e3
d_longeron = 10.2 * 20
d_batten = 8. * 20
d_diagonal = 2. * 20
# longeron
zoom = [l_longeron / 2, d_longeron / 2, d_longeron / 2]
trans = [l_longeron / 2, l_batten / 3**0.5, 0.]
rot = np.eye(3)
pos4d = [compose(trans, rot, zoom)]

# batten
zoom = [l_batten / 2, d_batten / 2, d_batten / 2]
trans = [0., 123.8, 400.]
rot = euler2mat(np.pi / 2, - np.pi / 6, 0, 'szxz')
pos4d.append(compose(trans, rot, zoom))

# diagonal1
zoom = [(l_longeron**2 + l_batten**2)**0.5 / 2.,
        d_diagonal / 2., d_diagonal / 2.]
trans = [l_longeron / 2., 123.8, 400.]
rot = euler2mat(np.deg2rad(90. - 34.03), -np.pi / 6, 0, 'szxz')
pos4d.append(compose(trans, rot, zoom))

# diagonal2
rot = euler2mat(np.deg2rad(90. + 34.03), -np.pi / 6, 0, 'szxz')
pos4d.append(compose(trans, rot, zoom))

onestage = Parallel(elem_class=Rod, elem_pos=pos4d)

from mayavi import mlab
from marxs.visualization.mayavi import plot_object
fig = mlab.figure()
obj = plot_object(onestage, viewer=fig)
