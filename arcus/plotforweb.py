import os
import numpy as np
import copy
import astropy.units as u
import astropy.table

import marxs
import marxs.optics
from marxs.source import PointSource, JitterPointing

from marxs.math.pluecker import h2e
import marxs.visualization.utils

import arcus.arcus as arcus
from arcus.arcus import rowland, aper, mirror, gas, catsupport, det, jitter_sigma

# output hardware and roland tori
with open('../website/arcus.json', 'w') as f:
    arcus.arcus_for_plot.plot(format="threejs", outfile=f)
    arcus.rowland.plot(theta0=3., thetaarc=3.4, phi0=-1, phiarc=1., outfile=f, format='threejs')
    arcus.rowlandm.plot(theta0=3., thetaarc=3.4, phi0=-1, phiarc=1., outfile=f, format='threejs')

# Now do different simulations through the same hardware
EQPegAspec = Table.read('../inputdata/EQPegA_flux.tbl', format='ascii', names=['energy', 'flux'])
# restrict table to ARCUS energy range
EQPegAspec = EQPegAspec[(EQPegAspec['energy'] > 0.25) & (EQPegAspec['energy'] < 1.5)]


energy = [(21.6 * u.Angstrom).to(u.keV, equivalencies=u.spectral()).value,
          (33.7 * u.Angstrom).to(u.keV, equivalencies=u.spectral()).value,
          EQPegAspec]
filename = ['o7.json', 'c6.json','eqpeg.json']
colcolor = ['order', 'order', 'energy']

for i in range(3):
    star = PointSource(coords=(23., 45.), flux=5., energy=energy[i])
    pointing = JitterPointing(coords=(23., 45.), jitter=jitter_sigma)
    photons = star.generate_photons(exposuretime=2000)
    photons = pointing(photons)
    p1 = arcus.arcus_extra_det(photons[0 : len(photons) / 2])
    p2 = arcus.arcus_extra_det_m(photons[len(photons) / 2: -1])
    p = astropy.table.vstack([p1, p2])
    marxs.visualization.mayavi.plot_rays(d, viewer=fig)
    pos1 = marxs.visualization.format_saved_positions(arcus.arcus.keeppos)
    pos2 = marxs.visualization.format_saved_positions(arcus.arcus.keepposm)
    pos = np.vstack([pos1, pos2])
    ind = p['probability'] > 0.
    with open('../website/' + filename[i], 'w') as f:
                ind = p['probability'] > 0.
                threejsjson.plot_rays(d[ind, :, :], scalar=np.abs(photons['order'][ind]), outfile=f)


