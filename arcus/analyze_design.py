'''Functions in this module help to summarize properties of design

This module is meant to provide functions that help to summarize properties
from different instrument designs. For example, one might run different
simulations with different focal lengths or grating placements.

In their current form, most of these functions make additional assumptions
on the structure of the photon lists. These are explained below.
Functions might be generalized and moved into marxs itself,
but until that happens, they live in this module.

Assumptions:
Each photon list represents one instrument design.
The input spectrum is the same for every simulation. It consists of
discrete wavelengths points and for each wavelength it has N_PHOT
photons (where N_PHOT is recorded as a keyword in the meta information
of the table).
'''

from __future__ import print_function

import numpy as np
from astropy.table import Table
import astropy.units as u

from marxs.analysis import gratings as anagrat


header_vals = ['BLAZE', 'D_CHAN', 'MAX_F', 'CIRCLE_R', 'TORUS_R']
'''Keywords in meta that describe the instrument design.

These keywords are printed in progress messages on the terminal and
copied to a results table.
(All keywords are expected to hold float values).
'''


### Quick helper functions ###
def between(data, val):
    return (data > val[0]) & (data < val[1])


def in_start_delta(data, start, delta):
    return between(data, [start, start + delta])


def zero_pos(p, coord='circ_phi'):
    ind = (p['order'] == 0) & np.isfinite(p[coord])
    return np.mean(p[coord][ind])


def angle_covered_by_CCDs(p, n_ccds=8):
    return n_ccds * 49.5 / p.meta['CIRCLE_R']


def add_phifolded(p):
    # calculate which angles can be covered
    p.meta['phi_0'] = zero_pos(p)

    # phi at the point excatly half-way between the two channels, the "middle"
    p.meta['phi_m'] = p.meta['phi_0'] - np.arcsin(p.meta['D_CHAN'] / 2 / p.meta['CIRCLE_R'])

    # Distance fomr middle to 0
    p.meta['dphi_0m'] = abs(p.meta['phi_0'] - p.meta['phi_m'])

    # make new column "distance from phi_m"
    p['phi_folded'] = np.abs(p['circ_phi'] - p.meta['phi_m'])


def get_wave(photons):
    p_wave = photons['wave', 'probability'].group_by('wave')
    return p_wave.groups.keys['wave'].to(u.Angstrom,
                                           equivalencies=u.spectral())


### Analysis functions

def aeff(p):
    p_wave = p['wave', 'probability'].group_by('wave')
    aeff = p_wave['wave', 'probability'].groups.aggregate(np.sum)
    aeff['probability'] *= p.meta['A_GEOM'] * 4  / p.meta['N_PHOT']
    aeff.rename_column('probability', 'area')
    aeff['area'].unit = u.cm**2
    return aeff


def calc_resolution(p, orders):
    '''Calculate the resolving power from a simulation result

    Parameters
    ----------
    photons : photon list
    orders : np.array of shape (M,)

    Returns
    -------
    resolvingpower : np.array of shape (N,)
        Resolving power averages of all dispersed orders
    res_out : np.array of shape (M, N)
        Resolving power per order
    prob_out : np.array of shape (M, N)
        Weighting factor for the power received in every order.
        This is not normalized and the sum over all orders for a particular
        wavelength will be less than 1.
    '''
    p_wave = p['wave', 'probability', 'circ_phi', 'order'].group_by('wave')
    res_out = np.zeros((len(p_wave.groups), len(orders)))
    prob_out = np.zeros_like(res_out)
    for i, group in enumerate(p_wave.groups):
        res, pos, std = anagrat.resolvingpower_from_photonlist(group, orders, col='circ_phi',
                                                               zeropos=p.meta['phi_0'])
        res_out[i, :] = res
        # Now get how important every order is for order-weighted plots
        for j, o in enumerate(orders):
            prob_out[i, j] = (group['probability'][group['order'] == o]).sum()
    # Normalize prob_out
    probs = prob_out / prob_out.sum(axis=1)[:, None]
    ind = (orders != 0)
    resolvingpower = np.nansum(res_out[:, ind] * probs[:, ind], axis=1)
    return resolvingpower, res_out, prob_out


def ccd8zeroorder(p):
    '''One possible function to select the "optimal" detector position

    Parameters
    ----------
    p: photon list
        needs to have phi_folded in it
    '''
    ang8 = angle_covered_by_CCDs(p, n_ccds=8)
    binwidth = angle_covered_by_CCDs(p, n_ccds=0.1)
    # Look at region +- 8 CCDS from the zeros order, because we definitely
    # want that in the range.
    # Don't go +/- 8 CCDs, but a little less, so zeroth order is never
    # exactly on the edge of detector
    bins = np.arange(p.meta['dphi_0m'] - (ang8 - binwidth), p.meta['dphi_0m'] + (ang8 + binwidth), binwidth)
    hist, bin_edges = np.histogram(p['phi_folded'], weights=p['probability'], bins=bins)
    signal = np.cumsum(hist)
    signal8 = signal[80:] - signal[:-80]
    return bins[np.argmax(signal8)]


def make_det_scenarios(p):
    pdisp = p[p['order'] < 0]

    # Scenario 1: All photons
    det_scenarios = [{'phi_start': 0., 'phi_stop': .2,
                      'scenario_name': 'all_photons'}]

    # Scenario 2: Maximize dispersed photons
    phistart = ccd8zeroorder(pdisp)
    det_scenarios.append({'phi_start': phistart,
                          'phi_stop': phistart + angle_covered_by_CCDs(p),
                          'scenario_name': '8 CCDs'})

    # Scenario 3: Maximize dispersed O VII photons
    po7 = pdisp[between(pdisp['wave'], [19.9, 22.5])]
    phistart = ccd8zeroorder(po7)
    det_scenarios.append({'phi_start': phistart,
                          'phi_stop': phistart + angle_covered_by_CCDs(p),
                          'scenario_name': '8 CCDs, max O VII'})
    return det_scenarios


### Functions that deal with files and loops

def load_prepare_table(filename):
    p = Table.read(filename)
    ind = np.isfinite(p['circ_phi']) & np.isfinite(p['order']) & (p['probability'] > 0)
    p = p[ind]
    add_phifolded(p)
    p['wave'] = p['energy'].to(u.Angstrom, equivalencies=u.spectral())
    return p


def summarize_file(filename, orders, make_det_scenarios=make_det_scenarios):
    p = load_prepare_table(filename)
    out = []
    for det in make_det_scenarios(p):
        pdet = p[between(p['phi_folded'],
                         [det['phi_start'], det['phi_stop']])]
        aeff_sum = aeff(pdet)
        resolvingpower, res_out, probs = calc_resolution(pdet, orders)
        out.append({'filename': filename,
                    'aeff': aeff_sum['area'],
                    'res': resolvingpower,
                    'res_per_order': res_out, 'prob_per_order': probs})
        out[-1].update(det)
        for k in header_vals:
            out[-1][k] = p.meta[k]
    return out, get_wave(p)
