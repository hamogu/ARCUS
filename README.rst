Ray-tracing ARCUS with the MARXS package
----------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

This package contains the setup code to ray-trace ARCUS.


License
-------

This project is Copyright (c) MIT/Hans Moritz Guenther and licensed under the terms of the GNU GPL v3+ license. See the licenses folder for more information.


Install
-------

First, make sure you have the following prerequisites:

- astropy (`Installation instructions <http://www.astropy.org for installation
  instructions>`_)
- tranforms3d (``pip install transforms3d``)
- MARXS (`Installation instruction
  <http://marxs.readthedocs.io/en/latest/install.html>`_ - you can ignore the
  section about classic MARXS with is not required for ARCUS simulations).

Second, get the git repository of ARCUS input data from bitbucket
(Note that at this time, that repository is not public. If you are part
of the ARCUS collaboration, contact Adam for access.).
The following command will download that repository into your current
working directory

    git clone git@bitbucket.org:arcus_xray/caldb-inputdata.git

Then, place a file called ``arcus.cfg`` in either the working directory
where you want to run the simulations or in ``~/.astropy/config/``.
The file must have the following form::

    [data]
    ## Path to git repository for caldb-inputdata repository
    caldb_inputdata = /path/to/your/caldb-inputdata

    [verbosity]
    ## Level of verbosity
    verbose = 1

    ## Default logging level
    logging_level = INFO

where you need to set the path to location where you placed the input data
repository in the step above.

Last, we need to install the arcus module. Since this module is still changing
frequently, I recommend to get most current version (not just the last released
version) by doing::

    git clone https://github.com/hamogu/ARCUS.git
    python setup.py install

Every time you need a new version of the data files, go into the 
``caldb-inputdata`` directory and type ``git pull``. Similarly, to update
the arcus module do ``git pull`` followed by ``pytohn setup.py install``.

Using ARCUS
-----------
More detailed instructions will come. For now, please look at Moritz' notebooks
(they all have a button to show/hide the code, so you can see how the
calculations are done) and the code itself in the git repository.
