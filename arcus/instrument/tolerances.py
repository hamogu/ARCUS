from numpy.random import rand, randn

from marxs.missions.mitsnl.catgrating import InterpolateEfficiencyTable


class OrderSelectorWavy(InterpolateEfficiencyTable):
    '''Add a random number to blaze angle before looking up Ralf Table

    In the lab, it seems that the grating bars are not exactly
    perpendicular to the "surface" of the grating. This class adds
    a random number drawn from a Gaussian distribution to the blaze angle
    before looking up the grating efficiency and selecting the order.

    Parameters
    ----------
    wavysigma : float
        Sigma of Gaussian distribution (in radian)
    '''
    def __init__(self, wavysigma, **kwargs):
        self.sigma = wavysigma
        super().__init__(**kwargs)

    def probabilities(self, energies, pol, blaze):
        return super().probabilities(energies, pol, blaze + self.sigma * randn(len(blaze)))


class OrderSelectorTopHat(InterpolateEfficiencyTable):
    '''Add a random number to blaze angle before looking up Ralf Table

    In the lab, it seems that the grating bars are not exactly
    perpendicular to the "surface" of the grating. This class adds
    a random number drawn from a Gaussian distribution to the blaze angle
    before looking up the grating efficiency and selecting the order.

    Parameters
    ----------
    tophatwidth : float
        width of tophat function (in radian)
    '''
    def __init__(self, tophatwidth, **kwargs):
        self.tophatwidth = tophatwidth
        super().__init__(**kwargs)

    def probabilities(self, energies, pol, blaze):
        return super().probabilities(energies, pol, blaze + self.tophatwidth * (rand(len(blaze)) - 0.5))
