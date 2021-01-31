'''This module contains functions to write Arcus CALDB files.'''
import glob
from datetime import datetime
from os.path import join as pjoin
import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
from marxs.math.utils import h2e
from .arcus import DetCamera, defaultconf
from .utils import TagVersion
from . import conf


def versionstring():
    return datetime.now().isoformat().replace('T', '').replace('-', '').replace(':', '')[:14]


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
    det = DetCamera(defaultconf)

    tab = Table(names=['CCDID', 'XPIX', 'YPIX', 'XPXL', 'YPXL',
                       'XWIDTH', 'YWIDTH',
                       'FOC0', 'FOCN', 'FOCX', 'READDIR'],
                dtype=[int, int, int, float, float, float, float,
                       '3f4', '3f4', '3f4', 'a2']
                )
    for c in ['XPIX', 'YPIX']:
        tab[c].unit = u.pixel
    for c in ['XPXL', 'YPXL', 'XWIDTH', 'YWIDTH', 'FOC0', 'FOCX']:
        tab[c].unit = u.mm

    for e in det.elements:
        row = {'CCDID': e.id_num,
               'XPIX': e.npix[0],
               'YPIX': e.npix[1],
               'XPXL': e.pixsize,
               'YPXL': e.pixsize,
               'XWIDTH': np.linalg.norm(h2e(e.geometry['v_y'])) * 2,
               'YWIDTH': np.linalg.norm(h2e(e.geometry['v_z'])) * 2,
               'FOC0': h2e(e.geometry['center']
                           - e.geometry['v_y']
                           - e.geometry['v_z']),
               'FOCN': h2e(e.geometry['e_x']),
               'FOCX': h2e(e.geometry['e_y']),
               'READDIR': '+y',
               }
        if tab is None:
            tab = Table(row)  # first row
        else:
            tab.add_row(row)  # later rows
    tab.meta['CALTYPE'] = 'FOCPLANE'
    tab.meta['VERSION'] = versionstring()
    tab.meta['INSTRUME'] = 'CCD'
    tab.meta['FILTER'] = 'none'
    tab.meta['EXTNAME'] = 'CALTYPE'
    tab = tagversion(tab)
    tab.sort('CCDID')
    tab.write(pjoin(conf.caldb_inputdata, 'fits', 'focplane.fits'),
              overwrite=True)


def combine_henke_reflectivity_solid_mirror(pattern, outfile):
    '''Combine tables downloaded from CXRO

    The CXRO website allows only to change one parameter at a time,
    for example change the energy of a photon for a fixed inclination
    angle to get the refelctivity. To build up a 2D distribution of
    reflectivity as a function of energy and angle, several different
    tables have to be downloaded.  This function combines several
    downloaded files.

    It only works for files from
    http://henke.lbl.gov/optical_constants/mirror2.html
    because it parses the meta data in the first line of the files and the
    format of that metadata is different for e.g. multi-layer mirrors.

    Parameters
    ----------
    pattern : string
        pattern for glob with wildcards to identify all input files
    outfile : string
        Filename and path of the output file.

    '''
    taball = []
    for filename in glob.iglob(pattern):
        t = Table.read(filename, format='ascii.no_header', data_start=2,
                       names=['energy', 'reflectivity'], guess=False)
        with open(filename) as f:
            firstline = f.readline().strip()
        material, roughness, polarization, angle = firstline.split(', ')
        t['angle'] = float(angle[:-3])
        taball.append(t)
    tab = vstack(taball)
    # Sort correctly for read-in routine
    tab.sort(['energy', 'angle'])
    # Make defined column order
    tab = tab['energy', 'angle', 'reflectivity']
    tab['energy'] = tab['energy'] / 1000
    tab['energy'].unit = u.keV
    tab['angle'].unit = u.degree
    tab.meta['Origin'] = 'CXRO'
    tab.meta['url'] = 'http://henke.lbl.gov/optical_constants/mirror2.html'
    tab.meta['material'] = material
    tab.meta['roughness'] = roughness
    tab.meta['polarization'] = polarization
    tab.write(outfile, format='ascii.ecsv')


def combine_henke_reflectivity_bilayer_mirror(pattern, outfile):
    '''Combine tables downloaded from CXRO

    The CXRO website allows only to change one parameter at a time,
    for example change the energy of a photon for a fixed inclination
    angle to get the refelctivity. To build up a 2D distribution of
    reflectivity as a function of energy and angle, several different
    tables have to be downloaded.  This function combines several
    downloaded files.

    It only works for files from
    http://henke.lbl.gov/optical_constants/bilayer.html
    because it parses the meta data in the first line of the files and the
    format of that metadata is different for e.g. multi-layer mirrors.

    Parameters
    ----------
    pattern : string
        pattern for glob with wildcards to identify all input files
    outfile : string
        Filename and path of the output file.

    '''
    taball = []
    for filename in glob.iglob(pattern):
        t = Table.read(filename, format='ascii.no_header', data_start=2,
                       names=['energy', 'reflectivity', 'transmission'],
                       guess=False)
        with open(filename) as f:
            firstline = f.readline().strip()
        material, polarization = firstline.split(', ')
        material, angle = material.split(' at ')
        t['angle'] = float(angle.replace('deg', ''))
        taball.append(t)
    tab = vstack(taball)
    # Sort correctly for read-in routine
    tab.sort(['energy', 'angle'])
    # Make defined column order
    tab = tab['energy', 'angle', 'reflectivity']
    tab['energy'] = tab['energy'] / 1000
    tab['energy'].unit = u.keV
    tab['angle'].unit = u.degree
    tab.meta['Origin'] = 'CXRO'
    tab.meta['url'] = 'http://henke.lbl.gov/optical_constants/bilayer.html'
    tab.meta['material'] = material
    tab.meta['polarization'] = polarization
    tab.write(outfile, format='ascii.ecsv')
