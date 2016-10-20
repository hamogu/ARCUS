import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u

from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

a = Table.read('../results/aeff.fits')

Ageometric = 5738

x = (a['energy'] * u.keV).to(u.Angstrom , equivalencies=u.spectral())


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, np.sum(a['fA'][:, :20],axis=1) * Ageometric, 'k', label='all orders')
for i in np.arange(-7, 0):
    ax.plot(x, a['fA'][:, i + 20] * Ageometric, label='order {0}'.format(i))

ax.legend()
ax.set_xlabel('wavelength [$\AA{}$]')
ax.set_ylabel('$A_{eff}$ [cm$^2]$')
ax.set_xlim([10, 60])

ax.annotate('order -1 hits chip gap', xy=(46, 20), xytext=(35, 800),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=20
            )

ax.annotate('order -4 falls off chip', xy=(34, 100), xytext=(15, 300),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=20)

fig2 = plt.figure()
ax = fig2.add_subplot(111)

for start, stop in zip([14, 13, 13, 12, 12, 10], [19, 20, 21, 20, 21, 24]):
    ax.plot(x, np.sum(a['CCD_ID'].data[:, start : stop] * Ageometric, axis=1), label=str(stop-start))

ax.legend(title='# CCDs')
ax.set_xlabel('wavelength [$\AA{}$]')
ax.set_ylabel('$A_{eff}$ [cm$^2]$')
ax.set_xlim([10, 50])
