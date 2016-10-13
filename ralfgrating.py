import numpy as np
from openpyxl import load_workbook


class EfficiencyFile(object):
    '''Select grating order from a probability distribution in a data file.
    '''
    def __init__(self, filename, sheetname, wave, orders, data):
        wb = load_workbook(filename)
        sheet = wb[sheetname]

        self.wave = np.array([d[0].value for d in sheet[wave]])
        self.orders = np.array([d.value for d in [e for e in sheet[orders]][0]])
        rows = [f for f in sheet[data]]
        self.prob = np.empty((len(rows), len(rows[0])))
        for i, r in enumerate(rows):
            for j, d in enumerate(r):
                self.prob[i, j] = d.value

        self.energy = 1.2398419292004201 / self.wave  # nm to keV
        # Probability to end up in any order
        self.totalprob = np.sum(self.prob, axis=1)
        # Cumulative probability for orders, normalized to 1.
        self.cumprob = np.cumsum(self.prob, axis=1) / self.totalprob[:, None]

    def __call__(self, energies, *args):
        orderind = np.empty(len(energies), dtype=int)
        ind = np.empty(len(energies), dtype=int)
        for i, e in enumerate(energies):
            ind[i] = np.argmin(np.abs(self.energy - e))
            orderind[i] = np.min(np.nonzero(self.cumprob[ind[i]] > np.random.rand()))
        return self.orders[orderind], self.totalprob[ind]

Efficiency191 = EfficiencyFile('../Efficiencies.xlsx', 'model 1.91deg', 'C12:C62', 'D11:P11', 'D12:P62')
