#    Copyright (C) 2021  Massachusetts Institute of Technology
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
from os.path import join as pjoin
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

import astropy.units as u
from arcus.reduction import arfrmf, ogip


@u.quantity_input(energy=u.keV, equivalencies=u.spectral())
def sig_ccd(energy):
    return np.interp(energy.to(u.keV, equivalencies=u.spectral()),
                     arfrmf.ccdfwhm['energy'] * u.keV,
                     arfrmf.ccdfwhm['FWHM'] * u.eV) / (2 * np.sqrt(2 * np.log(2)))


@u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
def osip_factor(chan_mid_nominal, o_nominal, o_true, sig_ccd, osip_table):
    if callable(osip_table):
        ohw = osip_table(chan_mid_nominal)
    else:
        ohw = osip_table
    ohw = np.broadcast_to(ohw, (2, len(chan_mid_nominal)), subok=True)

    # Need to exlicitly say if this is in energy or in wavelength space,
    # so there is a lot to(u.eV) in here
    osiptab = ohw.to(u.eV, equivalencies=u.spectral())
    # dE is distance from the energy of the nominal order
    dE = chan_mid_nominal.to(u.eV, equivalencies=u.spectral()) * ((o_true / o_nominal) - 1)
    lower_bound = dE - osiptab[0, :]
    upper_bound = dE + osiptab[1, :]
    scale = sig_ccd(chan_mid_nominal * o_true / o_nominal).to(u.eV, equivalencies=u.spectral())
    osip_factor = norm.cdf(upper_bound / scale) - norm.cdf(lower_bound / scale)
    return osip_factor


def apply_osip(arfrmfpath, outpath, name, order, osip_half_width, sig_ccd,
               offset_orders=[-1, 0, 1]):

    for t in offset_orders:
        m = order + t
        # No contamination by zeroth order or by orders on the other side
        # of the zeroth order
        if m != 0 and (np.sign(order) == np.sign(m)):
            try:
                filename = pjoin(arfrmfpath,
                                 f'{name}_' +
                                 arfrmf.filename_from_meta(filetype='arf',
                                                           ARCCHAN='all',
                                                           ORDER=order))
                arf = ogip.ARF.read(filename)
            except FileNotFoundError:
                print(f'Skipping order: {order}, trueorder: {m}, name: {name}')
                continue
            energies = 0.5 * (arf['ENERG_LO'] + arf['ENERG_HI'])
            osip_fac = osip_factor(energies, order, m, sig_ccd,
                                   osip_half_width)
            arf['SPECRESP'] = osip_fac * arf['SPECRESP']
            arf['OSIPFAC'] = osip_fac
            for tab in [arf]:
                tab.meta['INSTRUME'] = f'ORDER_{order}'
                if hasattr(osip_half_width, '__name__'):
                    tab.meta['OSIP'] = osip_half_width.__name__
                elif hasattr(osip_half_width, 'value'):
                    tab.meta['OSIP'] = 2 * osip_half_width.value
                else:
                    tab.meta['OSIP'] = True
                tab.meta['RFLORDER'] = f'{m}'
                tab.meta['CCDORD'] = f'{order}'
            # some times there is no overlap and all elements become 0
            if not np.all(arf['SPECRESP'] == 0):
                arf.write(pjoin(outpath,
                                f'{name}_' +
                                arfrmf.filename_from_meta('arf', **arf.meta)),
                          overwrite=True)


def p90(energy):
    return norm.interval(.9)[1] * sig_ccd(energy)


def osip_touch_factory(order, scale=1):
    def osip_touch(energy):
        dE = energy.to(u.keV, equivalencies=u.spectral()) * (((abs(order) + 1) / abs(order)) - 1)
        inter = interp1d(energy, dE / 2 * scale)
        return inter(energy) * dE.unit
    return osip_touch


def plot_osip(ax, grid, o, osip, color=None):
    en = (grid / np.abs(o)).to(u.keV, equivalencies=u.spectral())
    if callable(osip):
        ohw = osip(en)
    else:
        ohw = osip
    ohw = np.broadcast_to(ohw, (2, len(en)), subok=True)

    ax.plot(grid, en, label=o, c=color)
    ax.fill_between(grid, en - ohw[0, :], en + ohw[1, :],
                    color=color, alpha=.2)


def plot_mixture(ax, grid, osip, o=-5):
    en = (grid / np.abs(o)).to(u.keV, equivalencies=u.spectral())
    cm = osip_factor(en, o, o, sig_ccd, osip)
    cplus = osip_factor(en, o, o - 1, sig_ccd, osip)
    cminus = osip_factor(en, o, o + 1, sig_ccd, osip)
    ax.plot(grid, cm, label='true order')
    ax.plot(grid, cplus, label='upper contam')
    ax.plot(grid, cminus, label='lower contam')
    ax.plot(grid, cm + cplus + cminus, label='sum')
