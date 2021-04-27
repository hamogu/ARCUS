#    Copyright (C) 2021  Massachusetts Institute of Technology
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
from os.path import join as pjoin
from glob import glob
from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np
import astropy.units as u
from astropy.table import Table

from arcus.instrument.ccd import CCDRedist
from arcus.reduction import arfrmf, osip


def test_make_arf_osip():
    '''integration test: Generate and ARF and then apply an OCIP to it'''
    with TemporaryDirectory() as tmpdirname:
        # -5 is last, because we need that for the tests below
        for o in [-4, -6, -5]:
            arf = arfrmf.mkarf([23, 24, 25, 26] * u.Angstrom, o)
            arf = arfrmf.tagversion(arf)

            basearf = pjoin(tmpdirname,
                            arfrmf.filename_from_meta('arf', **arf.meta))
            arf.write(basearf)

        osipp = osip.FixedFractionOSIP(0.7, ccd_redist=CCDRedist())
        osipp.apply_osip_all(tmpdirname, tmpdirname, [-5])
        # there are three confused arfs now (to be read below)
        assert len(glob(pjoin(tmpdirname, '*-*-*'))) == 3

        # Check plots are created, but don't check content.
        # Not sure how to compare pdfs without too much work.
        assert len(glob(pjoin(tmpdirname, '*.pdf'))) > 0
        # Check something resonable is in the version file
        with open(pjoin(tmpdirname, 'README.md')) as f:
            readme = f.read()
        assert 'DATE' in readme
        assert str(datetime.now())[:9] in readme

        # using astropy.table here to be independent of the implementation of
        # ARF in arcus.reduction.ogip
        barf = Table.read(basearf, format='fits')


        # For debugging: Keep list of all filenames in varible
        globres = glob(pjoin(tmpdirname, '*'))

        # Using glob to get filename, to be independent of
        # filename_from_meta
        arfcenter = Table.read(glob(pjoin(tmpdirname, '*-5*-5*'))[0],
                               format='fits')
        assert np.isclose(arfcenter.meta['OSIPFAC'], 0.7)
        arfup = Table.read(glob(pjoin(tmpdirname, '*-5*-4*'))[0],
                           format='fits')
        arfdown = Table.read(glob(pjoin(tmpdirname, '*-5*-6*'))[0],
                             format='fits')
        # In this setup, some area falls between extraction regions
        assert np.all(barf['SPECRESP'] >
                      (arfcenter['SPECRESP'] +
                       arfup['SPECRESP'] +
                       arfdown['SPECRESP']))
