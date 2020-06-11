import os
import logging
import subprocess
from astropy.table import Table
import astropy.units as u
from . import conf

hash_displayed = False

# subprocess uses fork internally, which makes a child process
# that essentially makes a copy of everything that python has
# in memory, essentially doubling the memory footprint of python
# at that point. For long running scripts with big simulations that
# can be enough to run out of memory.
# So, just get this info once during the first import and save in
# module-level variables.
git_hash = subprocess.check_output(["git", "describe", "--always"],
                                   cwd=conf.caldb_inputdata)[:-1]

git_info = subprocess.check_output(['git', 'show', '-s', '--format=%ci',
                                    git_hash],
                                   cwd=conf.caldb_inputdata)


class DataFileFormatException(Exception):
    pass


def get_git_hash():
    return git_hash


def string_git_info():
    return 'hash: {} (commited on {})'.format(git_hash, git_info[:-15])


def log_tab_metadata(dirname, filename, tab):
    '''Print information about loaded files to standard out.

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)
    valuename : string
        Name of the column that hold the value
    '''
    global hash_displayed
    if conf.verbose > 0:
        if not hash_displayed:
            logging.info('data files in {}: version {}'.format(conf.caldb_inputdata, string_git_info()))
            hash_displayed = True
    if conf.verbose > 1:
        logging.info('Loading data from {0}/{1}'.format(dirname, filename))
    if (conf.verbose > 2) and ('keywords' in tab['meta']):
        for k in tab.meta['keywords']:
            logging.info('    {:<15} = {}'.format(k, tab.meta['keywords'][k]))
    if conf.verbose > 3:
        for k in tab.meta:
            if k != 'keywords':
                logging.info('{}: {}'.format(k, tab.meta[k]))


def load_number(dirname, filename, valuename):
    '''Get a single number from an ecsv input file

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)
    valuename : string
        Name of the column that hold the value

    Returns
    -------
    val : float or `astropy.units.Quantity`
        If the unit of the column is set, returns a `astropy.units.Quantity`
        instance, otherwise a plain float.
    '''
    tab = Table.read(os.path.join(conf.caldb_inputdata, dirname,
                                  filename + '.csv'), format='ascii.ecsv')
    log_tab_metadata(dirname, filename, tab)
    if len(tab) != 1:
        raise DataFileFormatException('Table {} contains more than one row of data.'.format(filename))
    else:
        if tab[valuename].unit is None:
            return tab[valuename][0]
        else:
            return u.Quantity(tab[valuename])[0]


def load_table(dirname, filename):
    '''Get a table from an ecsv input file

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)

    Returns
    -------
    val : `astropy.table.Table`
    '''
    tab = Table.read(os.path.join(conf.caldb_inputdata, dirname,
                                  filename + '.csv'), format='ascii.ecsv')
    log_tab_metadata(dirname, filename, tab)
    return (tab)


def load_table2d(dirname, filename):
    '''Get a 2d array from an ecsv input file.

    In the table file, the data is flattened to a 1d form.
    The first two columns are x and y, like this:
    The first column looks like this with many duplicates:
    [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3, ...].
    Column B repeats like this: [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3, ...].

    All remaining columns are data on the same x-y grid, and the grid
    has to be regular.


    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)

    Returns
    -------
    tab : `astropy.table.Table`
        Table as read in. Useful to access units or other meta data.
    x, y : `astropy.table.Column`
        Unique entries in first and second column
    dat : np.array
        The remaining outputs are np.arrays of shape (len(x), len(y))
    '''
    tab = Table.read(os.path.join(conf.caldb_inputdata, dirname,
                                  filename + '.csv'), format='ascii.ecsv')
    log_tab_metadata(dirname, filename, tab)

    x = tab.columns[0]
    y = tab.columns[1]
    n_x = len(set(x))
    n_y = len(set(y))
    if len(x) != (n_x * n_y):
        raise DataFileFormatException('Data is not on regular grid.')

    x = x[::n_y]
    y = y[:n_y]
    coldat = [tab[d].data.reshape(n_x, n_y) for d in tab.columns[2:]]

    return tab, x, y, coldat
