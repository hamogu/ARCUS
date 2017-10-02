'''This module contains functions to write Arcus CALDB files.'''

from datetime import datetime
from os.path import join as pjoin
import numpy as np
from astropy.table import Table
import astropy.units as u
from marxs.math.utils import h2e
from .arcus import DetTwoStrips, defaultconf
from .utils import TagVersion
from . import conf


def versionstring():
    return datetime.now().isoformat().replace('T','').replace('-', '').replace(':', '')[:14]


def focplane(tagversion={}):
    '''Write location of CCD detectors to CALDB file.

    Parameters
    ----------
    tagversion : dict
        Keywords for `arcus.utils.TagVersion`. See Examples.

    Example
    -------
    Write focplane.fits to the CALDB:

    >>> focplane(tagversion={'creator': 'Guenther', 'origin': 'MIT'})
    '''
    tagversion = TagVersion(**tagversion)
    det = DetTwoStrips(defaultconf)

    tab = Table(names=['CCDID', 'XPIX', 'YPIX', 'XPXL', 'YPXL', 'XWIDTH', 'YWIDTH',
                       'FOC0', 'FOCN', 'FOCX', 'READDIR'],
                dtype=[str, int, int, float, float, float, float,
                       '3f4', '3f4', '3f4', str]
                )
    for c in ['XPIX', 'YPIX']:
        tab[c].unit = u.pixel
    for c in ['XPXL', 'YPXL', 'XWIDTH', 'YWIDTH', 'FOC0', 'FOCX']:
        tab[c].unit = u.mm

    for e in det.elements:
        row = {'CCDID': e.name.replace('CCD ', ''),
               'XPIX': e.npix[0],
               'YPIX': e.npix[1],
               'XPXL': e.pixsize,
               'YPXL': e.pixsize,
               'XWIDTH': np.linalg.norm(h2e(e.geometry('v_y'))) * 2,
               'YWIDTH': np.linalg.norm(h2e(e.geometry('v_z'))) * 2,
               'FOC0': h2e(e.geometry('center') - e.geometry('v_y') - e.geometry('v_z')),
               'FOCN': h2e(e.geometry('e_x')),
               'FOCX': h2e(e.geometry('e_x')),
               'READDIR': '+y',
        }
        if tab is None:
            tab = Table(row) # first row
        else:
            tab.add_row(row) # later rows
    tab.meta['CALTYPE'] = 'FOCPLANE'
    tab.meta['VERSION'] = versionstring()
    tab.meta['INSTRUME'] = 'CCD'
    tab.meta['filter'] = 'none'
    tab = tagversion(tab)
    tab.write(pjoin(conf.caldb_inputdata, 'fits', 'focplane.fits'), overwrite=True)
