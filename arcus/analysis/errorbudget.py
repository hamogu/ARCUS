#    Copyright (C) 2022  Massachusetts Institute of Technology
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''A module to run full error budget simulations.
'''
import logging

from astropy.table import Table
import astropy.units as u
import numpy as np

from marxs.design.tolerancing import CaptureResAeff_CCDgaps
from arcus.instrument import Arcus, PerfectArcus
from arcus.instrument.arcus import reformat_randall_errorbudget
from arcus.defaults import DefaultSource, DefaultPointing


src = DefaultSource(energy=0.5 * u.keV)
pointing = DefaultPointing()
wave = np.array([15., 25., 37.]) * u.Angstrom

instrumfull = PerfectArcus(channels='1')
analyzer = CaptureResAeff_CCDgaps(A_geom=instrumfull.elements[0].area.to(u.cm**2),
                          dispersion_coord='circ_phi',
                          orders=np.arange(-12, 5),
                          aeff_filter_col='CCD')

logger = logging.getLogger(__name__)


def run_n_errorbudgets(align, conf, energies=wave, n=50, n_photons=200_000,
                       save_photon_lists=None):
    '''Run Arcus R/Aeff simulations for a particular set of alignment tolerances

    Parameters
    ----------
    align : list
        Error budget in the Randall-Smith form
    conf : dict
        Configuration dictionary. In particular, the alignment tolerance table in
        that dict determines the aligments for the runs.
    energies : list of astropy.units.Quantity
        List of energies or wavelength at which to run the simulations.
    n : int
        Number of simulations. The first simulation is always run with a perfect
        instrument for comparison, so ``n=50`` will get 49 simulations with random
        misalignments.
    n_photons : int
        Number of photons for each simulation.
    save_photon_lists : str
        If not `None` keep a copy of each photon list for debugging.
        String can be of the form `"/path/to/dir/backup_{e}_{n}.fits"`.

    Returns
    -------
    tab : `astropy.table.Table`
        Table with results. The first row is a run with a perfect instrument for
        comparison, the remaining rows are runs with random realizations of the
        alignment tolerances in `conf`.
    '''
    out = []

    for i in range(n):
        logger.info('Run tolerance budget: {}/{}'.format(i, n))

        reformat_randall_errorbudget(align, globalfac=None)
        conf['alignmentbudget'] = align

        if i == 0:
            arc = PerfectArcus(channels='1')
        else:
            arc = Arcus(channels='1', conf=conf)

        for e in energies:
            src.energy = e
            photons_in = src.generate_photons(n_photons * u.s)
            photons_in = pointing(photons_in)
            photons = arc(photons_in)
            if save_photon_lists is not None:
                photons.write(save_photon_lists.format(e=e, i=i))

            out.append(analyzer(photons))
            out[-1]['energy'] = e.to(u.keV, equivalencies=u.spectral()).value
            out[-1]['run'] = i

    tab = Table([{d: out[i][d].value
                  if isinstance(out[i][d], u.Quantity) else out[i][d]
                  for d in out[i]} for i in range(len(out))])

    tab['energy'].unit = u.keV
    tab['wave'] = tab['energy'].to(u.Angstrom, equivalencies=u.spectral())
    return tab