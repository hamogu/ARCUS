import numpy as np
import astropy.units as u
from scipy.interpolate import RectBivariateSpline
from marxs.base import SimulationSequenceElement
from marxs.optics import GlobalEnergyFilter
from marxs.optics.base import OpticalElement
from marxs.simulator import ParallelCalculated
from marxs.math.rotations import ex2vec_fix
from marxs.math.utils import e2h, h2e
import transforms3d

from .load_csv import load_table2d, load_number


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
        wave, theta, names, orders = load_table2d('gratings', 'efficiency')
        theta = theta.to(u.rad)
        # Order is int, we will never interpolate about order,
        # thus, we'll just have
        # len(order) 2d interpolations
        self.orders = -np.array([int(n[1:]) for n in names])
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
        photons['probability'][ind] *= np.exp(- (2 * np.pi * self.sigma / self.d)**2)**(photons['order'][ind]**2)
        return photons


def catsupportbars(photons):
    '''Metal structure that holds grating facets will absorb all photons
    that do not pass through a grating facet.

    We might want to call this L3 support ;-)
    '''
    photons['probability'][photons['facet'] < 0] = 0.
    return photons


catsupport = GlobalEnergyFilter(filterfunc=lambda e: load_number('gratings', 'L1support', 'transmission') *
                                load_number('gratings', 'L2support', 'transmission'))

class RectangularGrid(ParallelCalculated, OpticalElement):
    '''A collection of diffraction gratings on the Rowland torus.

    This class is similar to ``marxs.design.rowland.RectangularGrid`` but
    uses different axes.

    When initialized, it places elements in the space available on the
    Rowland circle, most commonly, this class is used to place grating facets.

    After generation, individual facet positions can be adjusted by hand by
    editing the attributes `elem_pos` or `elem_uncertainty`. See
    `marxs.simulation.Parallel` for details.

    After any of the `elem_pos`, `elem_uncertainty` or
    `uncertainty` is changed, `generate_elements` needs to be
    called to regenerate the facets on the GAS.

    Parameters
    ----------
    rowland : RowlandTorus
    d_element : float
        Size of the edge of a element, which is assumed to be flat and square.
        (``d_element`` can be larger than the actual size of the silicon
        membrane to accommodate a minimum thickness of the surrounding frame.)
    z_range: list of 2 floats
        Minimum and maximum of the x coordinate that is searched for an
        intersection with the torus. A ray can intersect a torus in up to four
        points. ``x_range`` specififes the range for the numerical search for
        the intersection point.
    x_range, y_range: lost of two floats
        limits of the rectangular area where gratings are placed.

    '''

    id_col = 'facet'

    def __init__(self, **kwargs):
        self.x_range = kwargs.pop('x_range')
        self.y_range = kwargs.pop('y_range')
        self.z_range = kwargs.pop('z_range')
        self.rowland = kwargs.pop('rowland')
        self.d_element = kwargs.pop('d_element')
        kwargs['pos_spec'] = self.elempos
        if 'parallel_spec' not in kwargs.keys():
            kwargs['parallel_spec'] = np.array([0., 0., 1., 0.])

        super(RectangularGrid, self).__init__(**kwargs)

    def elempos(self):

        n_x = int(np.ceil((self.x_range[1] - self.x_range[0]) / self.d_element))
        n_y = int(np.ceil((self.y_range[1] - self.y_range[0]) / self.d_element))

        # n_y and n_z are rounded up, so they cover a slighty larger range than y/z_range
        width_y = n_y * self.d_element
        width_x = n_x * self.d_element

        ypos = np.arange(0.5 * (self.y_range[0] - width_y + self.y_range[1] + self.d_element), self.y_range[1], self.d_element)
        xpos = np.arange(0.5 * (self.x_range[0] - width_x + self.x_range[1] + self.d_element), self.x_range[1], self.d_element)
        xpos, ypos = np.meshgrid(xpos, ypos)

        zpos = []
        for x, y in zip(xpos.flatten(), ypos.flatten()):
            zpos.append(self.rowland.solve_quartic(x=x, y=y, interval=self.z_range))

        return np.vstack([xpos.flatten(), ypos.flatten(), np.array(zpos), np.ones_like(zpos)]).T

    def calculate_elempos(self):
        '''Calculate the position of elements based on some algorithm.

        Returns
        -------
        pos4d : list of arrays
            List of affine transformations that bring an optical element centered
            on the origin of the coordinate system with the active plane in the
            yz-plane to the required facet position on the Rowland torus.
        '''
        pos4d = []

        xyzw = self.elempos()
        normals = self.get_spec('normal_spec', xyzw)
        parallels = self.get_spec('parallel_spec', xyzw, normals)

        for i in range(xyzw.shape[0]):
            rot_mat = ex2vec_fix(h2e(normals[i, :]), h2e(parallels[i, :]))
            pos4d.append(transforms3d.affines.compose(h2e(xyzw[i, :]), rot_mat, np.ones(3)))
        return pos4d
