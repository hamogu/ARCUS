# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
comments here
"""
# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa
# ----------------------------------------------------------------------------
import os
import configparser

__all__ = ['config']

config = configparser.ConfigParser()
confs_found = config.read(['arcus.cfg',
                           os.path.expanduser('~/.astropy/config/arcus.cfg')
                           ])
