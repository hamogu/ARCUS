'''Analyze a grid of monoenergetic simulations for R and Aeff
'''
import os
import glob
import numpy as np
from astropy.table import Table
import astropy.units as u
from marxs.analysis.gratings import resolvingpower_from_photonlist as respow

chan_name = ['1', '2', '1m', '2m']
orders = np.arange(-12, 1)
apertures = np.arange(4)


def check_meta_consistent(meta1, meta2):
    '''Check that the meta data between two simulations indicates consistency.

    This check compares a number of keywords (e.g. the version of marxs) to
    indicate if the simulations where run with consistent setups. Since the
    meta data is not a complete list of the simulation setup (e.g. the number
    of channels is not recorded there) this is necessarily be incomplete.

    '''
    for k in ['CREATOR', 'MARXSVER', 'ARCUSVER', 'ARCDATHA', 'SATELLIT',
              'COORDSYS', 'RA_PNT', 'DEC_PNT', 'ROLL_PNT',
              'RA_NOM', 'DEC_NOM', 'ROLL_NOM']:
        assert meta1[k] == meta2[k]


def check_energy_consistent(photons):
    assert np.allclose(photons['energy'], photons['energy'][0])


def rel_aeff_from_photonlist(photons, orders, col='order'):
    '''Calculate the fraction of photons that are detected in a specific order

    While written for calculating Aeff per order, this can be used with any
    discrete quantity, e.g. Aeff per CCD.

    The input photon list must contain every generated photon and cannot be
    pre-filtered!

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon event list
    orders : np.array
        Orders for which the effective area will be calculated
    col : string
        Column name for the order column.

    Returns
    -------
    rel_aeff : np.array
        Effective area relative to the geometric area of the aperture.

    '''

    n = len(photons)
    good = photons['CCD'] >= 0
    rel_aeff = np.zeros_like(orders, dtype=float)
    for i, o in enumerate(orders):
        rel_aeff[i] = (photons['probability'].data[good & (photons[col] == o)]).sum()
    return rel_aeff / n


def analyse_sim(photons, orders, apertures, reference_meta, conf):
    '''Get R and Aeff from a single photon event list

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon event list
    order : np.array
        list of orders to be analyzed
    apertures : list or None
        List of values used in ``photons['aperture']`` column.
        If ``None``, just use all values in the column.
    reference_meta : OrderedDict or None
        If not ``None`` check that the meta information of the photon
        list matches the meta information in ``reference_meta``
    conf : dict
        Arcus configuration (the zero order position for each channel
        is taken from this dict).

    Returns
    -------
    res : np.array
        Array of R values for each aperture and order, measured from
        photons that hit a CCD
    relaeff : np.array
        Relative Aeff for each aperture and order
    res_circ : np.array
        Array of R values for each aperture and order, as measured from
        the ``circ_phi`` column. For signal close to a CCD boundary, the
        detected photon distribution may be artificially narrow. This takes
        all photons into account to avoid that problem. In general it is
        better to use res (because the CCDs don't follow the Rowland circle
        exactly) but it's so close that ``res_circ`` can be used for those
        cases.
    '''
    if apertures is None:
        apertures = list(set(photons['aperture']))
    if reference_meta is not None:
        check_meta_consistent(photons.meta, reference_meta)
    check_energy_consistent(photons)

    res = np.zeros((len(apertures), len(orders)))
    res_circ = np.zeros((len(apertures), len(orders)))
    relaeff = np.zeros_like(res)
    for ia, a in enumerate(apertures):
        pa = photons[(photons['aperture'] == a) &
                     (photons['CCD'] >= 0) &
                     (photons['probability'] > 0)]
        zeropos = conf['pos_opt_ax'][chan_name[a]][0]
        res_a, pos_a, std_a = respow(pa, orders, zeropos=zeropos)
        res[ia, :] = res_a

        if 'circ_phi' in photons.colnames:
            pc = photons[(photons['aperture'] == a) &
                         np.isfinite(photons['circ_phi']) &
                         (photons['probability'] > 0)]
            zeropos = np.arcsin((conf['d'] - conf['pos_opt_ax'][chan_name[a]][0]) /
                                conf['rowland_detector'].r)
            res_c, pos_c, std_c = respow(pc, orders, col='circ_phi',
                                     zeropos=zeropos)
            res_circ[ia, :] = res_c
        paper = photons[photons['aperture'] == a]
        relaeff[ia, :] = rel_aeff_from_photonlist(paper, orders)
    return res, relaeff, res_circ


def aeffRfromraygrid(inpath, aperture, conf, outfile):
    '''Analyse a grid of simulations for R and Aeff

    inpath : string
        Path to the simulations grid
    aperture : `marxs.optics.aperture.BaseAperture`
        Aperture used for the simulation (the geometric opening error is
        taken from this)
    conf : dict
        Arcus configuration (the zero order position for each channel
        is taken from this dict).
    outfile : string
        File names to save output table
    '''
    rayfiles = glob.glob(os.path.join(inpath, '*.fits'))
    rayfiles.sort()
    r0 = Table.read(rayfiles[0])
    energies = np.zeros(len(rayfiles))
    res = np.zeros((len(rayfiles), len(apertures), len(orders)))
    res_circ = np.zeros_like(res)
    aeff = np.zeros_like(res)

    for ifile, rayfile in enumerate(rayfiles):
        obs = Table.read(rayfile)
        res_i, relaeff_i, res_circ_i = analyse_sim(obs, orders, apertures, r0.meta, conf)
        res[ifile, :, :] = res_i
        res_circ[ifile, :, :] = res_circ_i
        aeff[ifile, :, :] = relaeff_i
        energies[ifile] = obs['energy'][0]

    res_clean = np.minimum(res, res_circ)
    a_geom = u.Quantity([a.area.to(u.cm**2) for a in aperture.elements])
    aeff = aeff * a_geom[None, :, None]
    wave = (energies * u.keV).to(u.Angstrom, equivalencies=u.spectral())
    aeff_4 = aeff.sum(axis=1)
    res_4 = np.ma.masked_invalid(np.ma.average(res_clean,
                                               weights=aeff, axis=1))
    res_disp = np.ma.average(res_4[:, :-1],
                             weights=np.ma.masked_equal(aeff_4[:, :-1], 0),
                             axis=1)
    out = Table([energies, wave, res, res_circ, res_clean,
                 aeff, aeff_4, res_4, res_disp],
                names=['energy', 'wave', 'Rccd', 'Rcirc', 'R',
                       'Aeff', 'Aeff4', 'R4', 'R_disp'])
    out['energy'].unit = u.keV
    out['wave'].unit = u.Angstrom
    out['Aeff'].unit = u.cm**2
    out['Aeff4'].unit = u.cm**2
    out.meta = r0.meta
    for i, o in enumerate(orders):
        out.meta['ORDER_{}'.format(i)] = o
    out.write(outfile, overwrite=True)


def csv_per_order(infile, col, outfile):
    '''Rewrite one column in ``aeffRfromraygrid`` to csv file

    Turn one vector-valued (all orders in one cell) column into a
    csv table with one entry per cell.
    '''
    tab = Table.read(infile)
    outtab = Table(tab[col], names=['order_{0}'.format(o) for o in orders])
    outtab.add_column(tab['wave'], index=0)
    outtab.write(outfile, format='ascii.csv', overwrite=True)
