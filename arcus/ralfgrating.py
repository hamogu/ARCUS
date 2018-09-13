import numpy as np
import astropy.units as u
from scipy.interpolate import RectBivariateSpline, interp1d
import transforms3d
from marxs.base import SimulationSequenceElement
from marxs.optics import (GlobalEnergyFilter, CATGrating, FlatOpticalElement,
                          OrderSelector, FlatStack)
from marxs.simulator import Parallel
from marxs.math.utils import norm_vector, h2e, e2h
from marxs.optics.scatter import RandomGaussianScatter
from marxs import energy2wave


from .load_csv import load_table2d, load_number, load_table


class InterpolateRalfTable(object):
    '''Order Selector for MARXS using a specific kind of data table.

    The data table was given to me by Ralf. It contains simulated grating
    efficiencies in an Excel table.
    A short summary of this format is given here, to help reading the code.
    The table contains data in 3-dimenasional (wave, n_theta, order) space,
    flattened into a 2d table.

    - Row 1 + 2: Column labels. Not used here.
    - Column A: wavelength in nm.
    - Column B: blaze angle in deg.
    - Rest: data

    For each wavelength there are multiple blaze angles listed, so Column A
    contains
    many dublicates and looks like this: [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3, ...].
    Column B repeats like this: [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3, ...].

    Because the wave, theta grid is regular, this class can use the
    `scipy.interpolate.RectBivariateSpline` for interpolation in each 2-d slice
    (``order`` is an integer and not interpolated).

    Parameters
    ----------
    filename : string
        path and name of data file
    k : int
        Degree of spline. See `scipy.interpolate.RectBivariateSpline`.
    '''

    def __init__(self, k=3):
        tab, wave, theta, orders = load_table2d('gratings', 'efficiency')
        theta = theta.to(u.rad)
        # Order is int, we will never interpolate about order,
        self.orders = np.array([int(n) for n in tab.colnames[2:]])
        self.interpolators = [RectBivariateSpline(wave, theta, d, kx=k, ky=k) for d in orders]

    def probabilities(self, energies, pol, blaze):
        '''Obtain the probabilties for photons to go into a particular order.

        This has the same parameters as the ``__call__`` method, but it returns
        the raw probabilities, while ``__call__`` will draw from these
        probabilities and assign an order and a total survival probability to
        each photon.

        Parameters
        ----------
        energies : np.array
            Energy for each photons
        pol : np.array
            Polarization for each photon (not used in this class)
        blaze : np.array
            Blaze angle for each photon

        Returns
        -------
        orders : np.array
            Array of orders
        interpprobs : np.array
            This array contains a probability array for each photon to reach a
            particular order
        '''
        # convert energy in keV to wavelength in nm
        # (nm is the unit of the input table)
        wave = (energies * u.keV).to(u.nm, equivalencies=u.spectral()).value
        interpprobs = np.empty((len(self.orders), len(energies)))
        for i, interp in enumerate(self.interpolators):
            interpprobs[i, :] = interp.ev(wave, blaze)
        return self.orders, interpprobs

    def __call__(self, energies, pol, blaze):
        orders, interpprobs = self.probabilities(energies, pol, blaze)
        totalprob = np.sum(interpprobs, axis=0)
        # Cumulative probability for orders, normalized to 1.
        cumprob = np.cumsum(interpprobs, axis=0) / totalprob
        ind_orders = np.argmax(cumprob > np.random.rand(len(energies)), axis=0)

        return orders[ind_orders], totalprob


class RalfQualityFactor(SimulationSequenceElement):
    '''Scale probabilites of theoretical curves to measured values.

    All gratings look better in theory than in practice. This grating quality
    factor scales the calculated diffraction probabilities to the observed
    performance.
    '''

    def __init__(self, **kwargs):
        self.sigma = load_number('gratings', 'debyewaller', 'sigma')
        self.d = load_number('gratings', 'debyewaller', 'd')
        super(RalfQualityFactor, self).__init__(**kwargs)

    def process_photons(self, photons):
        ind = np.isfinite(photons['order'])
        photons['probability'][ind] = photons['probability'][ind] * np.exp(- (2 * np.pi * self.sigma / self.d)**2)**(photons['order'][ind]**2)
        return photons


def catsupportbars(photons):
    '''Metal structure that holds grating facets will absorb all photons
    that do not pass through a grating facet.

    We might want to call this L3 support ;-)
    '''
    photons['probability'][photons['facet'] < 0] = 0.
    return photons

L2relarea = load_number('gratings', 'L2support', 'transmission')
catsupport = GlobalEnergyFilter(filterfunc=lambda e: L2relarea)


class CATfromMechanical(Parallel):
    '''A collection of diffraction gratings on the Rowland torus.

    After any of the `elem_pos`, `elem_uncertainty` or
    `uncertainty` is changed, `generate_elements` needs to be
    called to regenerate the facets on the GAS.
    '''

    id_col = 'window'

    def stack(self, name):
        return np.vstack([self.data[name + 'X'].data,
                          self.data[name + 'Y'].data,
                          self.data[name + 'Z'].data]).T

    def __init__(self, **kwargs):
        self.channel = kwargs.pop('channel')
        self.conf = kwargs.pop('conf')
        self.data = load_table('gratings', 'facets')

        zoom = [[1, row['xsize'] / 2, row['ysize'] / 2] for row in self.data]
        trans = self.stack('')
        rot = np.stack([self.stack('NormN'),
                        self.stack('DispN'),
                        self.stack('GBarN')])
        pos4d = [transforms3d.affines.compose(trans[i, :],
                                              rot[:, i, :].T,
                                              zoom[i])
                 for i in range(len(self.data))]

        mirr = np.eye(4)
        if 'm' in self.channel:
            mirr[0, 0] = 1
        else:
            mirr[0, 0] = -1
        if '2' in self.channel:
            mirr[1, 1] = -1

        pos4d = [np.dot(mirr, p) for p in pos4d]

        # Shift pos4d from focal point to center of grating petal
        # Ignore rotations at this point
        pos4d = np.array(pos4d)
        centerpos = np.eye(4)
        centerpos[:, 3] = pos4d.mean(axis=0)[:, 3]
        centerpos_inv = np.linalg.inv(centerpos)
        kwargs['pos4d'] = np.dot(centerpos, kwargs['pos4d'])
        pos4d = np.array([np.dot(centerpos_inv, p) for p in pos4d])

        # Now get centers for grating windows
        # could be more clever with binning and unique etc.
        # but easiest here is to assume that all window numbers are in use
        # That way, we also know that the numbering with id will match.
        # We check numbers with assert.
        assert set(self.data['SPO_MM_num']) == set(np.arange(self.data['SPO_MM_num'].max() + 1))
        assert np.all(np.diff(self.data['SPO_MM_num']) >= 0)  # ordered increasing

        windowpos = []
        gratingpos = []
        id_start = []
        d_grat = []

        for i in np.arange(self.data['SPO_MM_num'].max() + 1):
            ind = self.data['SPO_MM_num'] == i
            winpos = np.eye(4)
            winpos[:, 3] = pos4d[ind, :, :].mean(axis=0)[:, 3]
            winpos_inv = np.linalg.inv(winpos)
            windowpos.append(winpos)
            id_start.append(kwargs.get('id_num_offset', 0) +
                            self.data['facet_num'][ind][0])
            grat_pos = [np.dot(winpos_inv, pos4d[j, :, :]) for j in ind.nonzero()[0]]
            gratingpos.append(grat_pos)
            d_grat.append(list(self.data['period'][ind]))

        kwargs['elem_pos'] = windowpos
        kwargs['elem_class'] = CATWindow
        kwargs['elem_args'] = {}
        # This is the elem_args for the second level (the CAT gratings).
        # We need a list of dicts. Each window will then get one dict
        # which itself is a dict of lists
        # currenty, 'd' is the only parameter passed down that way
        # but e.g. orderselector could be treated the same way
        # kwargs['elem_args']['elem_args'] = [{'d': d} for d in d_grat]
        kwargs['elem_args']['elem_pos'] = gratingpos
        kwargs['elem_args']['id_num_offset'] = id_start
        super(CATfromMechanical, self).__init__(**kwargs)

    def generate_elements(self):
        super(CATfromMechanical, self).generate_elements()
        for e in self.elements:
            e.generate_elements()


globalorderselector = InterpolateRalfTable()
'''Global instance of an order selector to use in all CAT gratings.

As long as the efficiency table is the same for all CAT gratings, it makes
sense to define that globaly. If every grating had its own independent
order selector, we would have to read the selector file a few hundred times.
'''
l1relativearea = load_number('gratings', 'L1support', 'transmission')
l1transtab = load_table('gratings', 'L1transmission')
l1transmission = interp1d(l1transtab['energy'].to(u.keV, equivalencies=u.spectral()),
                          l1transtab['transmission'])


class CATGratingL1(CATGrating):
    '''A CAT grating representing only the L1 structure

    This is treated independently of the CAT grating layer itself
    although the two gratings are not really in the far-field limit.
    CAT gratings of this class determine (statistically) if a photon
    passes through the grating bars or the L1 support.
    The L1 support is simplified as solid Si layer of 4 mu thickness.
    '''
    l1transmission = l1transmission

    blaze_name = 'blaze_L1'
    order_name = 'order_L1'

    def specific_process_photons(self, photons, intersect,
                                 interpos, intercoos):
        catresult = super().specific_process_photons(photons, intersect, interpos, intercoos)

        # Now select which photons go through the L1 support and
        # set the numbers appropriately.
        # It is easier to have the diffraction calculated for all photons
        # and then re-set numbers for a small fraction here.
        # That, way, I don't have to duplicate the blaze calculation and no
        # crazy tricks are necessary to keep the indices correct.
        l1 = np.random.rand(intersect.sum()) > l1relativearea
        ind = intersect.nonzero()[0][l1]
        catresult['dir'][l1] = photons['dir'].data[ind, :]
        catresult['polarization'][l1] = photons['polarization'].data[ind, :]
        catresult['order_L1'][l1] = 0
        catresult['probability'][l1] = l1transmission(photons['energy'][ind])
        return catresult


class L2(FlatOpticalElement):
    '''L2 absorption and shadowing

    Some photons may pass through the CAT grating, but could then be
    absorbed by the L2 sidewalls. We treat this statistically
    by reducing the overall probability.
    I'm ignoring the effect that photons might scatter of the L2
    sidewall surface (those would be scattered away from the CCDs
    anyway).

    Note that this does not read the L2 from a file, but calcualtes it
    directly from the dimensions.
    '''
    bardepth = 0.5
    period = 0.916
    innerfree = 0.866

    def specific_process_photons(self, photons, intersect,
                                 interpos, intercoos):

        p3 = norm_vector(h2e(photons['dir'].data[intersect]))
        angle = np.arccos(np.abs(np.dot(p3, self.geometry['plane'][:3])))
        # Area is filled by L2 bars + area shadowed by L2 bars
        shadowarea = 3. * (self.period**2 - self.innerfree**2) + 2 * self.innerfree * 0.5 * np.sin(angle)
        shadowfraction = shadowarea /  (3. * self.period**2)

        return {'probability': 1. - shadowfraction}


class NonParallelCATGrating(CATGrating):
    '''CAT Grating where the angle of the reflective wall changes.

    This element represents a CAT grating where not all grating bar walls
    are perpendicular to the surface of the grating. This is only
    true for a ray through the center. The angle changes linearly with
    the distance to the center in the dispersion direction.
    Each grating bar has a fixed angle, i.e. no change of the direction
    happens along the grating bars (perpendicular to the dispersion direction).

    Parameters
    ----------
    d_blaze_mm : float
        Change in direction of the reflecting grating bar sidewall, which
        directly translates to a change in blaze angle [rad / mm].
    '''
    def __init__(self, **kwargs):
        self.d_blaze_mm = kwargs.pop('d_blaze_mm')
        super(NonParallelCATGrating, self).__init__(**kwargs)

    def blaze_angle_modifier(self, blazeangle, intercoos):
        '''
        Parameters
        ----------
        blazeangle : np.array
            Array of blaze angles for photons interacting with optical element
        intercoos : np.array
            intercoos coordinates for photons interacting with optical element
        '''
        return blazeangle + intercoos[:, 0] * self.d_blaze_mm

    def diffract_photons(self, photons, intersect, interpos, intercoos):
        '''Except for one line this is copied from marxs.optics.FlatGrating'''
        p = norm_vector(h2e(photons['dir'].data[intersect]))
        n = self.geometry('plane')[:3]
        l = h2e(self.geometry('e_groove'))
        # Minus sign here because we want n, l, d to be a right-handed coordinate system
        d = -h2e(self.geometry('e_perp_groove'))

        wave = energy2wave / photons['energy'].data[intersect]
        # calculate angle between normal and (ray projected in plane perpendicular to groove)
        # -> this is the blaze angle
        p_perp_to_grooves = norm_vector(p - np.dot(p, l)[:, np.newaxis] * l)
        # Use abs here so that blaze angle is always in 0..pi/2
        # independent of the relative orientation of p and n.
        blazeangle = np.arccos(np.abs(np.dot(p_perp_to_grooves, n)))

        ### This is the line that was inserted here ###
        blazeangle = self.blaze_angle_modifier(blazeangle,
                                               intercoos[intersect, :])

        m, prob = self.order_selector(photons['energy'].data[intersect],
                                      photons['polarization'].data[intersect],
                                      blazeangle)

        # The idea to calculate the components in the (d,l,n) system separately
        # is taken from MARX
        sign = self.order_sign_convention(p)
        p_d = np.dot(p, d) + sign * m * wave / self.d(intercoos[intersect, :])
        p_l = np.dot(p, l)
        # The norm for p_n can be derived, but the direction needs to be chosen.
        p_n = np.sqrt(1. - p_d**2 - p_l**2)
        # Check if the photons have same direction compared to normal before
        direction = np.sign(np.dot(p, n), dtype=np.float)
        if not self.transmission:
            direction *= -1
        dir = e2h(p_d[:, None] * d[None, :] + p_l[:, None] * l[None, :] + (direction * p_n)[:, None] * n[None, :], 0)

        return dir, m, prob, blazeangle


class GeneralLinearNonParallelCAT(NonParallelCATGrating):
    def __init__(self, **kwargs):
        self.blaze_center = kwargs.pop('blaze_center')
        super(GeneralLinearNonParallelCAT, self).__init__(**kwargs)

    def blaze_angle_modifier(self, blazeangle, intercoos):
        '''
        Parameters
        ----------
        blazeangle : np.array
            Array of blaze angles for photons interacting with optical element
        intercoos : np.array
            intercoos coordinates for photons interacting with optical element
        '''
        return blazeangle + self.blaze_center + intercoos[:, 0] * self.d_blaze_mm


l1orderselector = OrderSelector(orderlist=np.array([-4, -3, -2, -1, 0,
                                                    1, 2, 3, 4]),
                                p=np.array([0.006, 0.0135, 0.022, 0.028, 0.861,
                                            0.028, 0.022, 0.0135, 0.006]))


def l2diffraction(photons, intersect, interpos, intercoos):
    '''Very simple approximation of L2 diffraction effects.

    L2 is a hexagonal pattern, but at such a large spacing, that diffraction
    at higher orders can be safely neglected. The only thing that does
    happen is a slight broadening due to the single-slit function, and again,
    only the core of that matters. So, we simply approximate this here with
    simple Gaussian Scattering.
    '''
    wave = energy2wave / photons['energy'].data[intersect]
    sigma = 0.4 * np.arcsin(wave / 0.966)
    return np.random.normal(size=intersect.sum()) * sigma


class CATL1L2Stack(FlatStack):
    def __init__(self, **kwargs):
        kwargs['elements'] = [CATGrating,
                              CATGratingL1,
                              L2,
                              RandomGaussianScatter]
        kwargs['keywords'] = [{'order_selector': globalorderselector,
                               'd': 0.0002},
                              {'d': 0.005,
                               'order_selector': l1orderselector,
                               'groove_angle': np.pi / 2.},
                              {},
                              {'scatter': l2diffraction}]
        super().__init__(**kwargs)


class CATWindow(Parallel):

    id_col = 'facet'

    def __init__(self, **kwargs):
        kwargs['id_col'] = self.id_col
        kwargs['elem_class'] = CATL1L2Stack
        super().__init__(**kwargs)
