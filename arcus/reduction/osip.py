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
import string
from abc import ABC
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

import astropy.units as u
from astropy.table import Table
from arcus.reduction import arfrmf, ogip
from arcus.utils import OrderColor
from arcus.reduction.arfrmf import tagversion

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

logger = logging.getLogger(__name__)

__all__ = ['sig_ccd', 'OSIPBase',
           'FixedWidthOSIP', 'FixedFractionOSIP', 'FractionalDistanceOSIP',
           ]

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


class OSIPBase(ABC):
    '''Modify ARF files to account for order-sorting effects

    This is a base class that implements methods to modify ARF files
    to account for order-sorting. Different diffraction orders fall on
    the same physical space on the CCD. The CCD energy resolution can
    then be used to assign photons to specific grating
    orders. However, since the CCD energy resolution is finite, this
    assingment is not perfect. Some photons may fall outside of the
    nominal energy used for order sorting and thus be lost, other may
    fall onto the energy assgined tothe different order. This class
    modifies ARF files to correct for that.

    Parameters
    ----------
    offset_orders : list of int
        Offset from main order that is relevant.

    sig_ccd : callable
        Function that return the width the Gaussian sigma of the CCD
        resolution, given an input energy.
    '''
    osip_description = 'Not implemented'
    '''String to be added to FITS header in OSIP keyword for ARFs written.'''

    def __init__(self, offset_orders=[-1, 0, 1], sig_ccd=sig_ccd):
        self.offset_orders = offset_orders
        self.sig_ccd = sig_ccd

    @u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
    def osip_tab(self, chan_mid_nominal, order):
        raise NotImplementedError

    @u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
    def osip_factor(self, chan_mid_nominal, o_nominal, o_true):
        if np.sign(o_nominal) != np.sign(o_true):
            return 0.

        osiptab = self.osip_tab(chan_mid_nominal, o_nominal)
        # Need to exlicitly say if this is in energy or in wavelength space,
        # so there is a lot to(u.eV) in here
        osiptab = osiptab.to(u.eV, equivalencies=u.spectral())
        # dE is distance from the energy of the nominal order
        dE = chan_mid_nominal.to(u.eV, equivalencies=u.spectral()) * ((o_true / o_nominal) - 1)
        lower_bound = dE - osiptab[0, :]
        upper_bound = dE + osiptab[1, :]
        scale = self.sig_ccd(chan_mid_nominal * o_true / o_nominal).to(u.eV, equivalencies=u.spectral())
        osip_factor = norm.cdf(upper_bound / scale) - norm.cdf(lower_bound / scale)
        return osip_factor

    def apply_osip(self, inputarf, outpath, order, outroot='',
                   overwrite=False):
        '''Modify an ARF to account for incomplete order sorting

        This function reads an input ARF file, which contains the
        effective area for a grating order in Arcus. For a given order
        sorting window, it then calculates what fraction of the
        photons is lost. For example, if the OSIP is chosen to contain
        the 90% energy fraction, then the new ARF values will be 0.9
        times the input ARF.

        If the ``order`` of the new ARF differs from the order of the
        input ARF, then the new ARF is representative of order
        confusion, e.g. is shows how many photons of order 4 are
        sorted into the order=5 grating spectrum.

        Parameters
        ----------
        inputarf : string
            Filename and path of input arf
        outpath : string
            Location where the output arfs are deposited
        order : int
            Nominal order. (The true order of the input arf is taken from the
            input arf header data.)
        outroot : string
            prefix for output filename
        overwrite : bool
            Overwrite existing files?
        '''
        arf = ogip.ARF.read(inputarf)
        try:
            arf['SPECRESP'] = arf['SPECRESP'] / arf['OSIPFAC']
            logger.info(f'{inputarf} already has OSIP applied, reverting ' +
                        'before applying new OSIP.')
        except KeyError:
            pass

        m = int(arf.meta['ORDER'])

        energies = 0.5 * (arf['ENERG_LO'] + arf['ENERG_HI'])
        osip_fac = self.osip_factor(energies / m * order, order, m)
        arf['SPECRESP'] = osip_fac * arf['SPECRESP']
        arf['OSIPFAC'] = osip_fac

        arf.meta['INSTRUME'] = f'ORDER_{order}'
        arf.meta['OSIP'] = self.osip_description
        arf.meta['TRUEORD'] = f'{m}'
        arf.meta['CCDORDER'] = f'{order}'
        # some times there is no overlap and all elements become 0
        if np.all(arf['SPECRESP'] == 0):
            logger.info(f'True refl order {m} does not contributed to ' +
                        f'CCDORDER {order}. ' +
                        'Writing ARF with all entries equal to zero.')
        os.makedirs(outpath, exist_ok=True)
        arf.write(pjoin(outpath, outroot +
                        arfrmf.filename_from_meta('arf', **arf.meta)),
                  overwrite=overwrite)

    def write_readme(self, outpath, outroot=''):
        '''Write README file to directory with ARFs

        Parameters
        ----------
        outpath : string
            Location where the output ARFs are deposited
        outroot : string
            prefix for output filename
        '''
        # Get a table so that I can tag it and get all the meta information
        tag = Table()
        tagversion(tag)
        # Format the meta information into a string
        tagstring = ''
        for k, v in tag.meta.items():
            if isinstance(v, str):
                tagstring += f'{k}: {v}\n'
            else:
                tagstring += f'{k}: {v[0]}   // {v[1]}\n'

        with open(pjoin(os.path.dirname(__file__),
                        'data', "osip_template.md")) as t:
            template = string.Template(t.read())

        output = template.substitute(tagversion=tagstring)
        with open(pjoin(outpath, outroot + "README.md"), "w") as f:
            f.write(output)

    def plot_osip(self, ax, grid, order, **kwargs):
        '''Plot banana plot with OSIP region marked.

        Parameters
        ----------
        ax : `matplotlib.axes._subplots.AxesSubplot`
            The axes into which the banana is plotted.
        grid : `~astropy.units.quantity.Quantity`
            Wavelength grid
        order : int
            Order number
        kwargs :
            Any other parameters are passed to ``plt.plot``
        '''
        grid = grid.to(u.Angstrom, equivalencies=u.spectral())
        en = (grid / np.abs(order)).to(u.keV, equivalencies=u.spectral())

        ohw = self.osip_tab(grid, order)

        line = ax.plot(grid, en, label=order, **kwargs)
        ax.fill_between(grid, en - ohw[0, :], en + ohw[1, :],
                        color=line[0].get_color(), alpha=.2,
                        label='__no_legend__')
        ax.set_xlabel(f'$m\\lambda$ [{grid.unit.to_string("latex_inline")}]')
        ax.set_ylabel('CCD energy [keV]')
        ax.set_xlim([grid.value.min(), grid.value.max()])
        ax.legend()
        ax.set_title('Order sorting regions')

    def plot_mixture(self, ax, grid, order):
        '''Plot relative contribution of main order and interlopers.

        Parameters
        ----------
        ax : `matplotlib.axes._subplots.AxesSubplot`
            The axes into which the lines are plotted.
        grid : `~astropy.units.quantity.Quantity`
            Wavelength grid
        order : int
            Order number
        '''
        grid = grid.to(u.Angstrom, equivalencies=u.spectral())
        ax.axhspan(1, 2, facecolor='r', alpha=.3, label='extractions overlap')
        en = (grid / np.abs(order)).to(u.keV, equivalencies=u.spectral())
        cm = self.osip_factor(en, order, order)
        ax.plot(grid, cm, label='main order', lw=2)
        for o in self.offset_orders:
            if o == 0:
                continue
            coffset = self.osip_factor(en, order, order + o)
            ax.plot(grid, coffset, label=f'contam {o}',
                    # Different linestyle to avoid overlapping lines in plot
                    ls={-1: '-', +1: ':'}[np.sign(o)]
                    )
            cm += coffset
        ax.plot(grid, cm, 'k', label='sum', lw=3)
        ax.set_xlabel(f'$m\\lambda$ [{grid.unit.to_string("latex_inline")}]')
        ax.set_ylabel('Fraction of photons in OSIP')
        ax.set_title(f'Order {order}')
        ax.set_xlim([grid.value.min(), grid.value.max()])
        ax.set_ylim(0, cm.max() * 1.05)
        ax.legend()

    def plot_summary(self, inputarf, orders, outpath, outroot=''):
        '''Write summary plot to directory with ARFs

        Parameters
        ----------
        inputarf : string
            Path to one input ARFs. The energy grid for the plot
            is taken from that ARF.
        outpath : string
            Location where the output ARFs are deposited
        outroot : string
            prefix for output filename
        '''
        arf = ogip.ARF.read(inputarf)
        grid = 0.5 * (arf['ENERG_LO'] + arf['ENERG_HI']).to(u.Angstrom,
                                    equivalencies=u.spectral())
        fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

        oc = OrderColor(max_order=np.max(np.abs(orders)))

        for order in orders:
            self.plot_osip(axes[0], grid, order, **oc(order))
        # pick the middle order for plotting purposes
        o_mid = orders[len(orders) // 2]
        self.plot_mixture(axes[1], grid, o_mid)
        fig.subplots_adjust(wspace=.3)
        fig.savefig(pjoin(outpath, outroot + 'OSIP_regions.pdf'),
                    bbox_inches='tight')

    def apply_osip_all(self, inpath, outpath, orders,
                       inroot='', outroot='', ARCCHAN='all',
                       overwrite=False):
        '''Apply OSIP to many arfs at once

        This routine iterates over orders and offset orders to produce
        arfs that describe the contamination due to insufficient order
        sorting.  When a single order is extracted (e.g. order 5), but
        the CCD resolution is insufficient to really separate the
        orders well, then some photons from order 4 and 6 might end up
        in the extracted spectrum. This function generates the
        appropriate arfs.

        Input arfs need to follow the arcus filename convention and
        all be located in the same directory. In addition to the ARFs
        with order-sorting applied, this method also places an
        overview plot (assuming matplotlbi is available) and a readme
        file in the output directory.

        Parameters
        ----------
        inpath : string
            Directory with input ARFs.
        outpath : string
            Location where the output ARFs are deposited
        orders : list of int
            Nominal CCD orders to be processed
        inroot : string
            prefix for input filename
        outroot : string
            prefix for output filename
        ARCCHAN : string
            Channel for Arcus
        overwrite : bool
            Overwrite existing files?
        '''
        for order in orders:
            for t in self.offset_orders:
                # No contamination by zeroth order or by orders on the other
                # side of the zeroth order
                if (order + t != 0) and (np.sign(order) == np.sign(order + t)):
                    inputarf = pjoin(inpath, inroot +
                                     arfrmf.filename_from_meta(filetype='arf',
                                                               ARCCHAN=ARCCHAN,
                                                               ORDER=order + t))
                    try:
                        self.apply_osip(inputarf, outpath, order,
                                        outroot=outroot, overwrite=overwrite)
                    except FileNotFoundError:
                        logger.info(f'Skipping order: {order}, offset: {t} ' +
                                    'because input arf not found')
                        continue

        if HAS_PLT:
            self.plot_summary(inputarf, orders, outpath, outroot)
        self.write_readme(outpath, outroot)


class FixedWidthOSIP(OSIPBase):
    '''Modify ARF files to account for order-sorting effects

    This is a base class that implements methods to modify ARF files
    to account for order-sorting. Different diffraction orders fall on
    the same physical space on the CCD. The CCD energy resolution can
    then be used to assign photons to specific grating
    orders. However, since the CCD energy resolution is finite, this
    assingment is not perfect. Some photons may fall outside of the
    nominal energy used for order sorting and thus be lost, other may
    fall onto the energy assgined tothe different order. This class
    modifies ARF files to correct for that.

    Parameters
    ----------
    halfwidth : `astropy.units.quantity.Quantity`
        Half-wdith of the order sorting region. The same width is used for all
        wavelength.
    offset_orders : list of int
        Offset from main order that is relevant.
    sig_ccd : callable
        Function that return the width the Gaussian sigma of the CCD
        resolution, given an input energy.
    '''
    def __init__(self, halfwidth, **kwargs):
        self.halfwidth = halfwidth
        super().__init__(**kwargs)

    @property
    def osip_description(self):
        return str(self.halfwidth)

    @u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
    def osip_tab(self, chan_mid_nominal, order):
        return np.broadcast_to(self.halfwidth,
                               (2, len(chan_mid_nominal)), subok=True)


class FixedFractionOSIP(OSIPBase):
    '''Modify ARF files to account for order-sorting effects

    This is a base class that implements methods to modify ARF files
    to account for order-sorting. Different diffraction orders fall on
    the same physical space on the CCD. The CCD energy resolution can
    then be used to assign photons to specific grating
    orders. However, since the CCD energy resolution is finite, this
    assingment is not perfect. Some photons may fall outside of the
    nominal energy used for order sorting and thus be lost, other may
    fall onto the energy assgined tothe different order. This class
    modifies ARF files to correct for that.

    Parameters
    ----------
    fraction : float
        Number (between 0 and 1) that determins which fraction of the CCD
        energy distriution should be covered by the order sorting regions.
        The same width is used for all
        wavelength.
    offset_orders : list of int
        Offset from main order that is relevant.
    sig_ccd : callable
        Function that return the width the Gaussian sigma of the CCD
        resolution, given an input energy.
    '''
    def __init__(self, fraction, **kwargs):
        self.fraction = fraction
        super().__init__(**kwargs)

    @property
    def osip_description(self):
        return 'OSIPFrac' + str(self.fraction)

    @u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
    def osip_tab(self, chan_mid_nominal, order):
        halfwidth = norm.interval(self.fraction)[1] * self.sig_ccd(chan_mid_nominal)
        return np.broadcast_to(halfwidth,
                               (2, len(chan_mid_nominal)), subok=True)


class FractionalDistanceOSIP(OSIPBase):
    '''Modify ARF files to account for order-sorting effects

    This is a base class that implements methods to modify ARF files
    to account for order-sorting. Different diffraction orders fall on
    the same physical space on the CCD. The CCD energy resolution can
    then be used to assign photons to specific grating
    orders. However, since the CCD energy resolution is finite, this
    assingment is not perfect. Some photons may fall outside of the
    nominal energy used for order sorting and thus be lost, other may
    fall onto the energy assgined tothe different order. This class
    modifies ARF files to correct for that.

    Parameters
    ----------
    fraction : float
        Fraction (between 0 and 1) of the space between orders that will be
        covered by the extration region. For a value of 1, order-sorting
        regions just touch and each photon will be assigned to exactly one
        order.
    offset_orders : list of int
        Offset from main order that is relevant.
    sig_ccd : callable
        Function that return the width the Gaussian sigma of the CCD
        resolution, given an input energy.
    '''
    def __init__(self, fraction=1., **kwargs):
        self.fraction = fraction
        super().__init__(**kwargs)

    @property
    def osip_description(self):
        return 'OSIPDist' + str(self.fraction)

    @u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
    def osip_tab(self, chan_mid_nominal, order):
        energy = chan_mid_nominal.to(u.keV, equivalencies=u.spectral())
        dE = energy * (((abs(order) + 1) / abs(order)) - 1)
        inter = interp1d(energy, dE / 2 * self.fraction)
        halfwidth = inter(energy) * dE.unit
        return np.broadcast_to(halfwidth,
                               (2, len(chan_mid_nominal)), subok=True)
