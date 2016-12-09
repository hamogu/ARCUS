
from copy import deepcopy
import transforms3d
import transforms3d.euler
import numpy as np
import arcus

arcus_orig = arcus.arcus_extra_det
arcus_orig_m = arcus.arcus_extra_det_m


def move_element(aff, element):
    # If this is a sequence or parallel, recurse down to the leaves
    if hasattr(element, 'elements'):
        for e in element.elements:
            move_element(aff, e)
    else:
        # Some elements are global and don't have to be moved, e.g. GobalEnergyFilter
        if hasattr(element, "pos4d"):
            element.pos4d = np.dot(aff, element.pos4d)


def move_boom(aff, arcus):
    # For three elements of arcus are part of the boom: aper, mirror, gas
    for i in range(3):
        move_element(aff, arcus.elements[i])


mission = deepcopy(arcus_orig)

aff = transforms3d.affines.compose([0, 100, 0], np.eye(3), np.ones(3))


from mayavi import mlab
from marxs.math.pluecker import h2e
import marxs.visualization.mayavi

fig = mlab.figure()
arcus_orig.elements[0].plot(format='mayavi', viewer=fig)
arcus.det_16.plot(format='mayavi', viewer=fig)

for alpha in np.arange(0, 360, 5.):
    R = transforms3d.euler.euler2mat(np.deg2rad(alpha), 0, 0, 'szxz')
    mission = deepcopy(arcus_orig)
    aff = transforms3d.affines.compose(np.zeros(3), R, np.ones(3))
    move_boom(aff, mission)
    mission.elements[0].plot(format='mayavi', viewer=fig)

import numpy as np
import astropy.table
from astropy.utils.metadata import enable_merge_strategies
from marxs import utils

from marxs.source import PointSource, JitterPointing
from marxs.analysis import resolvingpower_from_photonlist
import arcus

n_photons = 1e5

# define position and spectrum of source
#mysource = PointSource((0., 0.),
#                       energy={'energy': np.array([0.25, 1.7]),
#                               'flux': np.ones(2)},
#                       flux=1.)
mysource = PointSource((0., 0.), energy=0.5, flux=1.)
mypointing = JitterPointing(coords=(0., 0.), jitter=arcus.jitter_sigma)

# MARXS code is still missing the implementation to rescale the input spectrum to the
# effective area of the instrument.
# Thus, set total number of photons to simulate here by hand.
photons = mysource.generate_photons(n_photons / 2)
photons = mypointing(photons)
photons = arcus.arcus_extra_det(photons)

photonsm = mysource.generate_photons(n_photons / 2)
photonsm = mypointing(photonsm)
photonsm = arcus.arcus_extra_det_m(photonsm)


res, pos, std = resolvingpower_from_photonlist(photons, np.arange(-11, 1))
resm, posm, stdm = resolvingpower_from_photonlist(photonsm, np.arange(-11, 1))

with enable_merge_strategies(utils.MergeIdentical):
    allphot = astropy.table.vstack([photons, photonsm])

orders = np.arange(-11, 1)
angles = np.arange(0., 1., .1)

respow = np.zeros((angles.shape[0], orders.shape[0]))
respowm = np.zeros_like(respow)

for i in range(angles.shape[0]):
    R = transforms3d.euler.euler2mat(np.deg2rad(angles[i]), 0, 0, 'szxz')
    aff = transforms3d.affines.compose(np.zeros(3), R, np.ones(3))
    mypointing = JitterPointing(coords=(angles[i], 0.), jitter=arcus.jitter_sigma)
    mission = deepcopy(arcus.arcus_extra_det)
    move_boom(aff, mission)
    missionm = deepcopy(arcus.arcus_extra_det_m)
    move_boom(aff, missionm)

    pr = mysource.generate_photons(n_photons / 2)
    pr = mypointing(pr)
    pr = mission(pr)
    res, pos, std = resolvingpower_from_photonlist(pr, orders)
    respow[i, :] = res

    prm = mysource.generate_photons(n_photons / 2)
    prm = mypointing(prm)
    prm = missionm(prm)
    resm, posm, stdm = resolvingpower_from_photonlist(prm, orders)
    respowm[i, :] = resm


with enable_merge_strategies(utils.MergeIdentical):
    allpr = astropy.table.vstack([pr, prm])
