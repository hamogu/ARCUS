from __future__ import print_function
import numpy as np
from mayavi import mlab
from astropy.coordinates import SkyCoord
import marxs
from marxs import visualization
from marxs.visualization.mayavi import plot_object, plot_rays

from marxs.source import PointSource, FixedPointing, JitterPointing
import arcus
import boom

%matplotlib

n_photons = 1e3

# define position and spectrum of source
mysource = PointSource(coords=SkyCoord(0., 0., unit='deg'),
                       energy={'energy': np.array([0.25, 1.7]),
                               'flux': np.ones(2)},
                       flux=1.)
jitterpointing = JitterPointing(coords=SkyCoord(0., 0., unit='deg'),
                                jitter=arcus.jitter_sigma)
fixedpointing = FixedPointing(coords=SkyCoord(0., 0., unit='deg'))


photons = mysource.generate_photons(n_photons)
photons = jitterpointing(photons)
photons = arcus.arcus4(photons)

# origin of coordinate system is one of the focal points.
# center boom around mid-point between the two focal points.
angles = np.arange(0, 2. * np.pi/3., 0.1)
angles = np.array([0.])
hitrod = np.zeros_like(angles)
hitrod_prob = np.zeros_like(angles)

for i, angle in enumerate(angles):
    rot = np.array([[1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]])
    myboom = boom.ThreeSidedBoom(orientation=rot, position=[0, arcus.d, 0])
    photons['hitrod'] = False
    photons = myboom(photons)
    hitrod[i] = photons['hitrod'].sum()
    hitrod_prob[i] = photons['probability'][photons['hitrod']].sum()

fig = mlab.figure()
obj = plot_object(arcus.arcus4, viewer=fig)
obj = plot_object(boom0, viewer=fig)
posdat = visualization.utils.format_saved_positions(arcus.keeppos4)
rays = plot_rays(arcus.keeppos4, scalar=np.asarray(photons['hitrod'], dtype=float) * 2)


photons['wave'] = marxs.energy2wave / photons['energy'] * 1e7 # in ang
a01 = photons['aperture'] < 1.5
ind = np.isfinite(photons['order'])

i7 = photons['order'] == -7

### Zoom in on one region

n_photons = 5e5
import matplotlib.pyplot as plt

outn = np.histogram(photons['wave'][ind & a01 & i7], bins=20)
outnabs = np.histogram(photons['wave'][ind & a01 & i7 & ~photons['hitrod']], bins=outn[1])

spec = plt.hist(photons['wave'][ind & a01 & i7], weights=photons['probability'][ind & a01 & i7], bins=outn[1])
specabs = plt.hist(photons['wave'][ind & a01 & i7 & ~photons['hitrod']], weights=photons['probability'][ind & a01 & i7 & ~photons['hitrod']], bins=outn[1])

plt.plot(out[1][:-1], out[0] / out1[0])

mysource = PointSource(coords=SkyCoord(0., 0., unit='deg'),
                       energy={'energy': np.array([0.65, 0.652]),
                               'flux': np.ones(2)}, flux=1.)
jitterpointing = JitterPointing(coords=SkyCoord(0., 0., unit='deg'),
                                jitter=arcus.jitter_sigma)
fixedpointing = FixedPointing(coords=SkyCoord(0., 0., unit='deg'))
myboom = boom.ThreeSidedBoom(position=[0, arcus.d, 0])

photonj = mysource.generate_photons(n_photons)
photonj = jitterpointing(photonj)
photonj = arcus.arcus(photonj)
photonj = myboom(photonj)

photons = photonj

out1 = plt.hist(photons['wave'][ind & a01 & i7], bins=20)
out = plt.hist(photons['wave'][ind & a01 & i7 & ~photons['hitrod']], bins=out1[1])


s1 = spo.SPOChannelasAperture()
keeppos = KeepCol('pos')
instrum = Sequence(elements=[s1, arcus.gas_1, arcus.filtersandqe, arcus.det_16, arcus.projectfp],
                   postprocess_steps=[keeppos])
