# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
comments here
"""
# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:
    # For egg_info test builds to pass, put package imports here.
    # from .arcus import *
    import logging
    import configparser
    import os

    confparse = configparser.RawConfigParser({'verbose': 1})
    confs_found = confparse.read(['arcus.cfg',
                                  os.path.expanduser('~/.astropy/config/arcus.cfg')
                        ])

    class Conf(object):
        '''Object to hold configuration items.

        It takes aspects from the pure Python ConfigParser which has
        the ability to read several files and astropy's config objects
        which take defaults and can be changed by the user at run time.
        (The main problem with using the astropy config objects is that
        the always have default setting in testing, while the arcus
        module **requires** a path to data just to import successfully.
        '''
        pass

    conf = Conf()
    conf.caldb_inputdata = confparse.get('data', 'caldb_inputdata')
    try:
        conf.verbose = confparse.getint('verbosity', 'verbose')
    except (configparser.MissingSectionHeaderError, configparser.NoOptionError):
        conf.verbose = 1

    try:
        conf.logging_level = confparse.get('verbosity', 'logging_level')
    except (configparser.MissingSectionHeaderError, configparser.NoOptionError):
        conf.logging_level = "INFO"

    logging.basicConfig(level=conf.logging_level)
    logging.info('Reading configuration data from {}'.format(confs_found))

    from .arcus import *
