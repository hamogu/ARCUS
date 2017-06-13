import os
import logging
import subprocess
from astropy.table import Table
import astropy.units as u
from . import datapath

hash_displayed = False


class DataFileFormatException(Exception):
    pass


def get_git_hash():
    return subprocess.check_output(["git", "describe", "--always"],
                                   cwd=datapath)[:-1]


def string_git_info():
    githash = get_git_hash()
    date = subprocess.check_output(['git', 'show',  '-s',  '--format=%ci', githash],
                                   cwd=datapath)
    return 'hash: {} - commited on {}'.format(githash, date)


def log_tab_metadata(dirname, filename, tab, verbose):
    '''Print information about loaded files to standard out.

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)
    valuename : string
        Name of the column that hold the value
    verbose : integer
        Level of verbosity. 0 means no output.
    '''
    global hash_displayed
    if verbose > 0:
        if not hash_displayed:
            logging.info('data files in {}: version {}'.format(datapath, string_git_info()))
            hash_displayed = True
        logging.info('Loading data from {0}/{1}'.format(dirname, filename))
    if (verbose > 1) and ('keywords' in tab['meta']):
        for k in tab.meta['keywords']:
            logging.info('    {:<15} = {}'.format(k, tab.meta['keywords'][k]))
    if verbose > 2:
        for k in tab.meta:
            if k != 'keywords':
                logging.info('{}: {}'.format(k, tab.meta[k]))


def load_number(dirname, filename, valuename, verbose=False):
    '''Get a single number from an ecsv input file

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)
    valuename : string
        Name of the column that hold the value
    verbose : integer
        Level of verbosity. 0 means no output.

    Returns
    -------
    val : float or `astropy.units.Quantity`
        If the unit of the column is set, returns a `astropy.units.Quantity`
        instance, otherwise a plain float.
    '''
    tab = Table.read(os.path.join(datapath, dirname,
                                  filename + '.csv'), format='ascii.ecsv')
    log_tab_metadata(dirname, filename, tab, verbose)
    if len(tab) != 1:
        raise DataFileFormatException('Table {} contains more than one row of data.'.format(filename))
    else:
        if tab[valuename].unit is None:
            return tab[valuename][0]
        else:
            return u.Quantity(tab[valuename])[0]


def load_table(dirname, filename, verbose=False):
    '''Get a table from an ecsv input file

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)
    verbose : bool
        If ``True`` output info when loading file.

    Returns
    -------
    val : `astropy.table.Table`
    '''
    tab = Table.read(os.path.join(datapath, dirname,
                                  filename + '.csv'), format='ascii.ecsv')
    log_tab_metadata(dirname, filename, tab, verbose)
    return (tab)


def load_table2d(dirname, filename, verbose=False):
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
    verbose : bool
        If ``True`` output info when loading file.

    Returns
    -------
    x, y : `astropy.table.Column`
    colnames : list
        List of names of the other columns (which hold the data)
    dat : np.array
        The remaining outputs are np.arrays of shape (len(x), len(y))
    '''
    tab = Table.read(os.path.join(datapath, dirname,
                                  filename + '.csv'), format='ascii.ecsv')
    log_tab_metadata(dirname, filename, tab, verbose)

    x = tab.columns[0]
    y = tab.columns[1]
    n_x = len(set(x))
    n_y = len(set(y))
    if len(x) != (n_x * n_y):
        raise DataFileFormatException('Data is not on regular grid.')

    x = x[::n_y]
    y = y[:n_y]
    colnames = tab.colnames[2:]
    coldat = [tab[d].data.reshape(n_x, n_y) for d in tab.columns[2:]]

    return x, y, colnames, coldat
