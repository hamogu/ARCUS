from __future__ import print_function

import inspect

import numpy as np
from transforms3d import affines, euler
from astropy.table import Table
from marxs.simulator import Parallel
from marxs.design.uncertainties import generate_facet_uncertainty as genfacun
from marxs.analysis.gratings import resolvingpower_from_photonlist as resol
from marxs.analysis.gratings import AnalysisError
''' Multiple energies at once or loop?
'''


def singletolerance(photons_in, instrum_before,
                    wigglefunc, wigglepars,
                    instrum_after, derive_result):
    photons = instrum_before(photons_in.copy())
    for i, pars in enumerate(wigglepars):
        print('Working on {}/{}'.format(i, len(wigglepars)))
        instrum = wigglefunc(pars)
        p_out = instrum(photons.copy())
        p_out = instrum_after(p_out)
        if 'det_x' in p_out.colnames:
            ind = np.isfinite(p_out['det_x']) & (p_out['probability'] > 0)
        else:
            ind = []
        derive_result(pars, p_out[ind], len(p_out))


class ParallelUncertainty(object):
    '''Apply uncertainties to `~marxs.simulator.Parallel` objects.

    This class is initialized with a Marxs simulation object
    (typically a `~marxs.simulator.Sequence`). When it is called,
    it applies specific misplacements to the elements of a
     `~marxs.simulator.Parallel` group. This class will descend through
    the object hierarchy of the input to find all matching
     `~marxs.simulator.Parallel` objects.

    Parameters
    ----------
    elements :  Marxs simulation object
        The object to start a search (typically a `marxs.simulator.Sequence`
        object.)
    parallel_class : class or None
        If this is ``None``, this will simply step though the levels of
        `~marxs.simulator.Sequence` objects until it finds any
        `~marxs.simulator.Parallel` object and then apply tolerances to
        all `~marxs.simulator.Parallel` objects on that level. If a class
        is given, instead tolerances will be applied to all instances
        of that class (not subclasses).

    '''
    def __init__(self, elements, parallel_class=None):
        self.elements = elements
        # cannot be used to find functions
        if not inspect.isclass(parallel_class):
            raise ValueError('{} is not a class.'.format(parallel_class))
        self.parallel_class = parallel_class

    def find_parallel(self, obj):
        '''Walk a hirachy of simulation elements to find `Parallel` elements.

        This walk down a hiracy of `marxs.simulator.Sequence` and
        `marxs.simulator.Parallel` objects to find the first level
        that contains an object derived from the `marxs.simulator.Parallel`
        class.

        Parameters
        ----------
        obj : Marxs simulation object
            The object to start a search (typically a `marxs.simulator.Sequence`
            object.)

        Returns
        -------
        out : list
            Python list of all `marxs.simulator.Parallel` objects in the level
            where the first `marxs.simulator.Parallel` was found.
        '''
        if isinstance(obj, Parallel):
            return [obj]
        elif hasattr(obj, 'elements'):
            a = []
            for e in obj.elements:
                a += self.find_parallel(e)
            return a
        else:
            return []

    def find_parallel_class(self, obj, cla):
        a = []
        if hasattr(obj, 'elements'):
            for e in obj.elements:
                a += self.find_parallel_class(e, cla)

        if type(obj) == cla:
            a += [obj]
        return a

    @property
    def parallels(self):
        # Yes, the self in this call is necessary and python will expand
        # it to (self, self).
        # This works, because find_parallel needs the first self (to call the
        # method recursively and the second self to look at the elements
        # of this object
        if self.parallel_class is None:
            return self.find_parallel(self.elements)
        else:
            return self.find_parallel_class(self.elements, self.parallel_class)

    def apply_uncertainty(self, e, parameters):
        '''Apply uncertainties to a `marxs.simulator.Parallel`.

        This method needs to be implemented by a derived class.
        '''
        raise NotImplementedError

    def __call__(self, parameters):
        for e in self.parallels:
            self.apply_uncertainty(e, parameters)
            e.generate_elements()
        return self.elements


class WiggleIndividualElements(ParallelUncertainty):
    '''Wiggle elements of a Parallel object individually,
    drawing their misplacement from a Gaussian distribution with the
    parameters given.
    '''
    def apply_uncertainty(self, e, parameters):
        e.elem_uncertainty = genfacun(len(e.elements), tuple(parameters[:3]),
                                      tuple(parameters[3:]))


class WiggleGlobalParallel(ParallelUncertainty):
    '''Move all elements of a Parallel object in the same way.'''
    def apply_uncertainty(self, e, parameters):
        e.uncertainty = affines.compose(parameters[:3],
                                        euler.euler2mat(parameters[3],
                                                        parameters[4],
                                                        parameters[5], 'sxyz'),
                                        np.ones(3))


class PeriodVariation(ParallelUncertainty):
    '''Randomly draw different grating periods for different gratings

    This class needs to be instantiated with the gratings, e.g.

    >>> dvar = PeriodVariation(elements, CATGrating) # doctest: +SKIP

    and the parameters are expected to have two components:
    center and sigma of a Gaussian distribution for grating contstant d.
    '''
    def apply_uncertainty(self, e, parameters):
        if not hasattr(e, '_d'):
            raise ValueError('Object {} does not have grating period `_d` attribute.'.format(e))
        e._d = np.random.normal(parameters[0], parameters[1])

    def __call__(self, parameters):
        for e in self.parallels:
            self.apply_uncertainty(e, parameters)
        return self.elements


class ScatterVariation(PeriodVariation):
    '''Change the scatter properties of an SPO'''
    def apply_uncertainty(self, e, parameters):
        e.inplanescatter = parameters[0]
        e.perpplanescatter = parameters[1]


class CaptureResAeff(object):
    '''Capture resolving power and effective area for a tolerancing simulation.

    Instances of this class can be called with a list of input parameters for
    a tolerancing simulation and a resulting photon list. The photon list
    will be analysed for resolving power and effective area in a number of
    relevant orders.
    Every instance of this object has a ``tab`` attribute and every time the
    instance is called it wadd one row of data to the table.

    Parameters
    ----------
    n_parameters : int
        Parameters will be stored in a vector-valued column called
        "Parameters". To generate this column in the correct size,
        ``n_parameters`` specifies the number of parameters.
    A_geom : number
        Geometric area of aperture for the simulations that this instance
        will analyze.
    on_detector_test : callable
        A function that can be called on a photon table and returns a
        boolean array with elements set to ``True`` for photons that hit
        the detector. (In order to calcu
    '''
    orders = np.arange(-10, 1)

    order_col = 'order'
    '''Column names for grating orders'''

    dispersion_coord = 'proj_x'
    '''Dispersion coordinate for
    `marxs.analysis.gratings.resolvingpower_from_photonlist`'''

    def __init__(self, n_parameters, Ageom=1):
        self.Ageom = Ageom
        form = '{}f4'.format(len(self.orders))
        self.tab = Table(names=['Parameters', 'Aeff0', 'Aeffgrat', 'Rgrat', 'Aeff', 'R'],
                         dtype=['{}f4'.format(n_parameters),
                                float, float, float, form, form]
                         )
        for c in ['Aeff', 'Aeffgrat', 'Aeff']:
            self.tab[c].unit = Ageom.unit

    def find_photon_number(self, photons):
        '''Find the number of photons in the simulation.

        This method simply returns the length of the photons list which
        works if it has not been pre-filtered in any step.

        Subclasses can implement other ways, e.g. to inspect the header for a
        keyword.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Photon list

        Returns
        -------
        n : int
            Number of photons
        '''
        return len(photons)

    def calc_result(self, filtered, n_photons):
        '''Calculate Aeff and R for an input photon list.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Photon list. This photon list should already be filtered
            and contain only "good" photons, i.e. photons that hit a
            detector and have a non-zero probability.
        n_photons : number
            In order to calculate the effective area, the function needs
            to calculate the fraction of detected photons and to do that
            the total number of simulated photons needs to be passed in.

        Returns
        -------
        result : dict
            Dictionary with per-order Aeff and R, as well as values
            summed over all grating orders.
        '''
        aeff = np.zeros(len(self.orders))
        disporders = self.orders != 0

        if len(filtered) == 0:
            res = np.nan * np.ones(len(self.orders))
            avggratres = np.nan
        else:
            try:
                res, pos, std = resol(filtered, self.orders,
                                      col=self.dispersion_coord,
                                      zeropos=None, ordercol=self.order_col)
            except AnalysisError:
                # Something did not work, e.g. too few photons to find zeroth order
                res = np.nan * np.ones(len(self.orders))

            for i, o in enumerate(self.orders):
                aeff[i] = filtered['probability'][filtered[self.order_col] == o].sum()
            aeff = aeff / n_photons * self.Ageom
            # Division by 0 causes more nans, so filter those out
            # Also, res is nan if less than 20 photons are detected
            # so we need to filter those out, too.
            ind = disporders & (aeff > 0) & np.isfinite(res)
            if ind.sum() == 0:  # Dispersed spectrum misses detector
                avggratres = np.nan
            else:
                avggratres = np.average(res[ind],
                                        weights=aeff[ind] / aeff[ind].sum())
        # The following lines work for an empty photon list, too.
        aeffgrat = np.sum(aeff[disporders])
        aeff0 = np.sum(aeff[~disporders])
        return {'Aeff0': aeff0, 'Aeffgrat': aeffgrat, 'Aeff': aeff,
                'Rgrat': avggratres, 'R': res}

    def __call__(self, parameters, photons, n_photons):
        out = self.calc_result(photons, n_photons)
        out['Parameters'] = parameters
        self.tab.add_row(out)
