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
import logging
import os
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

import astropy.units as u
from arcus.reduction import arfrmf, ogip
from arcus.utils import OrderColor

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

logger = logging.getLogger(__name__)


@u.quantity_input(energy=u.keV, equivalencies=u.spectral())
def sig_ccd(energy):
    '''Return the Gaussian sigma of the width of the CCD resolution

    Parameters
    ----------
    energy : `~astropy.units.quantity.Quantity`
        True photon energy.

    Returns
    -------
    sigma : `~astropy.units.quantity.Quantity`
        Width of the Gaussian
    '''
    return np.interp(energy.to(u.keV, equivalencies=u.spectral()),
                     arfrmf.ccdfwhm['energy'] * u.keV,
                     arfrmf.ccdfwhm['FWHM'] * u.eV) / (2 * np.sqrt(2 * np.log(2)))


@u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
def osip_factor(chan_mid_nominal, o_nominal, o_true, sig_ccd, osip):
    if np.sign(o_nominal) != np.sign(o_true):
        return 0.
    if callable(osip):
        ohw = osip(chan_mid_nominal)
    else:
        ohw = osip
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


def apply_osip(inputarf, outpath, order, osip, sig_ccd, outroot='',
               overwrite=False):
    '''Modify an ARF to account for incomplete order sorting

    This function reads an input ARF file, which contains the effective area
    for a grating order in Arcus. For a given order sorting window, it then
    calculates what fraction of the photons is lost. For example, if the OSIP
    is chosen to contain the 90% energy fraction, then the new ARF values will
    be 0.9 times the input ARF.

    If the ``order`` of the new ARF differs from the order of the input ARF,
    then the new ARF is representative of order confusion, e.g. is shows how
    many photons of order 4 are sorted into the order=5 grating spectrum.

    Paramters
    ---------
    inputarf : string
        Filename and path of input arf
    outpath : string
        Location where the output arfs are deposited
    order : int
        Base order. (The true order of the input arf is taken from the input
        arf header data.)
    osip : callable or `~astropy.units.quantity.Quantity`
        Half width of the order-sorting region. See `osip_factor` for details
        of the format.
    sig_ccd : callable
        Function that return the width the Gaussian sigma of the CCD
        resolution, given an input energy.
    outroot : string
        prefix for output filename
    overwrite : bool
        Overwrite existing files?
    '''
    arf = ogip.ARF.read(inputarf)
    try:
        arf['SPECRESP'] = arf['SPECRESP'] / arf['OSIPFAC']
        logger.info(f'{inputarf} already has OSIP applied, reverting before applying new OSIP.')
    except KeyError:
        pass

    m = int(arf.meta['ORDER'])

    energies = 0.5 * (arf['ENERG_LO'] + arf['ENERG_HI'])
    osip_fac = osip_factor(energies, order, m, sig_ccd, osip)
    arf['SPECRESP'] = osip_fac * arf['SPECRESP']
    arf['OSIPFAC'] = osip_fac

    arf.meta['INSTRUME'] = f'ORDER_{order}'
    if hasattr(osip, '__name__'):
        arf.meta['OSIP'] = osip.__name__
    elif hasattr(osip, 'value'):
        arf.meta['OSIP'] = 2 * osip.value
    else:
        arf.meta['OSIP'] = True
    arf.meta['RFLORDER'] = f'{m}'
    arf.meta['CCDORDER'] = f'{order}'
    # some times there is no overlap and all elements become 0
    if np.all(arf['SPECRESP'] == 0):
        logger.info(f'True refl order {m} does not contributed to CCDORDER {order}. No ARF file written for this case.')
    else:
        os.makedirs(outpath, exist_ok=True)
        arf.write(pjoin(outpath, outroot +
                        arfrmf.filename_from_meta('arf', **arf.meta)),
                  overwrite=overwrite)


def apply_osip_all(inpath, outpath, orders, osip, sig_ccd,
                   inroot='', outroot='', ARCCHAN='all',
                   offset_orders=[-1, 0, 1], overwrite=False):
    '''Calculate an ARF accounting for photons lost due to order-sorting.

    This routine iterates over orders and offset orders to produce arfs that
    describe the contamination due to insufficient order sorting.
    When a single order is extracted (e.g. order 5), but the CCD resolution
    is insufficient to really separate the orders well, then some photons
    from order 4 and 6 might end up in the extracted spectrum. This function
    generates the appropriate arfs.

    Paramters
    ---------
    inpath : string
        Directory with input ARFs.
    outpath : string
        Location where the output ARFs are deposited
    orders : list of int
        Order numbers to be processed
    osip : callable or `~astropy.units.quantity.Quantity` or list
        Half width of the order-sorting region. See `osip_factor` for details
        of the format.
        ``osip`` can also be a list, in which case it must have the same number
        of elements as ``orders`` and contain one callable or Quantity per
        order.
    sig_ccd : callable
        Function that return the width the Gaussian sigma of the CCD
        resolution, given an input energy.
    inroot : string
        prefix for input filename
    outroot : string
        prefix for output filename
    ARCCHAN : string
        Channel for Arcus
    offset_orders : list of int
        offset from main order
    overwrite : bool
        Overwrite existing files?
    '''
    if not isinstance(osip, list):
        osip = [osip] * len(orders)
    for order, thisosip in zip(orders, osip):
        for t in offset_orders:
            # No contamination by zeroth order or by orders on the other side
            # of the zeroth order
            if (order + t != 0) and (np.sign(order) == np.sign(order + t)):
                inputarf = pjoin(inpath, inroot +
                                 arfrmf.filename_from_meta(filetype='arf',
                                                           ARCCHAN=ARCCHAN,
                                                           ORDER=order))
                try:
                    apply_osip(inputarf, outpath, order + t, thisosip, sig_ccd,
                               outroot=outroot, overwrite=overwrite)
                except FileNotFoundError:
                    logger.info(f'Skipping order: {order}, offset: {t} because input arf not found')
                    continue

    if HAS_PLT:
        arf = ogip.ARF.read(inputarf)
        grid = arf['ENERG_LO'].to(u.Angstrom, equivalencies=u.spectral())
        fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

        oc = OrderColor(max_order=np.max(np.abs(orders)))

        for order, thisosip in zip(orders, osip):
            plot_osip(axes[0], grid, order, thisosip, **oc(order))
        # pick the middle order for plotting purposes
        o_mid = orders[len(orders) // 2]
        plot_mixture(axes[1], grid, o_mid, osip[o_mid])
        fig.subplots_adjust(wspace=.3)
        fig.savefig(pjoin(outpath, outroot + 'OSIP_regions.pdf'),
                    bbox_inches='tight')


def p90(energy):
    return norm.interval(.9)[1] * sig_ccd(energy)


def osip_touch_factory(order, scale=1):
    def osip_touch(energy):
        dE = energy.to(u.keV, equivalencies=u.spectral()) * (((abs(order) + 1) / abs(order)) - 1)
        inter = interp1d(energy, dE / 2 * scale)
        return inter(energy) * dE.unit
    return osip_touch


def plot_osip(ax, grid, order, osip, **kwargs):
    '''Plot banana plot with OSIP region marked.

    Parameters
    ----------
    ax : `matplotlib.axes._subplots.AxesSubplot`
        The axes into which the banana is plotted.
    grid : `~astropy.units.quantity.Quantity`
        Wavelength grid
    order : int
        Order number
    osip : number or callable
        OSIP specification, see `osip_factor` for the format
    kwargs :
        Any other parameters are passed to ``plt.plot``
    '''
    grid = grid.to(u.Angstrom, equivalencies=u.spectral())
    en = (grid / np.abs(order)).to(u.keV, equivalencies=u.spectral())
    if callable(osip):
        ohw = osip(en)
    else:
        ohw = osip
    ohw = np.broadcast_to(ohw, (2, len(en)), subok=True)

    line = ax.plot(grid, en, label=order, **kwargs)
    ax.fill_between(grid, en - ohw[0, :], en + ohw[1, :],
                    color=line[0].get_color(), alpha=.2,
                    label='__no_legend__')
    ax.set_xlabel(f'$m\\lambda$ [{grid.unit.to_string("latex_inline")}]')
    ax.set_ylabel('CCD energy [keV]')
    ax.set_xlim([grid.min(), grid.max()])
    ax.legend()
    ax.set_title('Order sorting regions')


def plot_mixture(ax, grid, order, osip, offset_orders=[-1, 1]):
    '''Plot relative contribution of main order and interlopers.

    Parameters
    ----------
    ax : `matplotlib.axes._subplots.AxesSubplot`
        The axes into which the lines are plotted.
    grid : `~astropy.units.quantity.Quantity`
        Wavelength grid
    order : int
        Order number
    osip : number or callable
        OSIP specification, see `osip_factor` for the format
    offset_orders : list of int
        Offset orders to be plotted
    '''
    grid = grid.to(u.Angstrom, equivalencies=u.spectral())
    ax.axhspan(1, 2, facecolor='r', alpha=.3, label='extractions overlap')
    en = (grid / np.abs(order)).to(u.keV, equivalencies=u.spectral())
    cm = osip_factor(en, order, order, sig_ccd, osip)
    ax.plot(grid, cm, label='main order', lw=2)
    for o in offset_orders:
        if o == 0:
            continue
        coffset = osip_factor(en, order, order + o, sig_ccd, osip)
        ax.plot(grid, coffset, label=f'contam {o}',
                # Different linestyle to avoid overlapping lines in plot
                ls={-1: '-', +1: ':'}[np.sign(o)]
                )
        cm += coffset
    ax.plot(grid, cm, 'k', label='sum', lw=3)
    ax.set_xlabel(f'$m\\lambda$ [{grid.unit.to_string("latex_inline")}]')
    ax.set_ylabel('Fraction of photons in OSIP')
    ax.set_title(f'Order {order}')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_ylim(0, cm.max() * 1.05)
    ax.legend()
