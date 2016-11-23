import numpy as np
import astropy.units as u
import astropy.table

import marxs
import marxs.optics
from marxs.source import PointSource, JitterPointing

import marxs.visualization.utils
from marxs.visualization import threejsjson

import arcus
from arcus import rowland, aper, mirror, gas, catsupport, det, jitter_sigma

# output hardware and roland tori
json1 = arcus.arcus_for_plot.plot(format="threejsjson")
rowlandkwargs = {'theta0': 3., 'thetaarc': 3.4, 'phi0': -1, 'phiarc': 1.}
json2 = arcus.rowland.plot(format="threejsjson", **rowlandkwargs)
json3 = arcus.rowlandm.plot(format="threejsjson", **rowlandkwargs)

with open('../website/arcus.json', 'w') as f:
    threejsjson.write(f, json1) # , json2, json3])

# Now do different simulations through the same hardware
EQPegAspec = astropy.table.Table.read('../inputdata/EQPegA_flux.tbl',
                                      format='ascii', names=['energy', 'flux'])
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
    photons = star.generate_photons(exposuretime=200)
    photons = pointing(photons)
    p1 = arcus.arcus_extra_det(photons[0: len(photons) / 2])
    p2 = arcus.arcus_extra_det_m(photons[len(photons) / 2: -1])
    p = astropy.table.vstack([p1, p2])
    pos1 = marxs.visualization.utils.format_saved_positions(arcus.keeppos)
    pos2 = marxs.visualization.utils.format_saved_positions(arcus.keepposm)
    pos = np.vstack([pos1, pos2])
    arcus.keeppos.data = []
    arcus.keepposm.data = []
    ind = p['probability'] > 0.
    if i < 2:
        s = np.abs(p['order'][ind])
    else:
        s = p['energy'][ind]
    json = [threejsjson.plot_rays(pos[ind, :, :], scalar=s)]
    json.extend(json1)
    with open('../website/' + filename[i], 'w') as f:
        threejsjson.write(f, json)
