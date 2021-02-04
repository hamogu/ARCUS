import re
from datetime import datetime
import subprocess

from marxs.base import MarxsElement
from . import version as vers
from . import config

# subprocess uses fork internally, which makes a child process
# that essentially makes a copy of everything that python has
# in memory, essentially doubling the memory footprint of python
# at that point. For long running scripts with big simulations that
# can be enough to run out of memory.
# So, just get this info once during the first import and save in
# module-level variables.
git_hash = subprocess.check_output(["git", "describe", "--always"],
                                   cwd=config['data']['caldb_inputdata'])[:-1]

git_info = subprocess.check_output(['git', 'show', '-s', '--format=%ci',
                                    git_hash],
                                   cwd=config['data']['caldb_inputdata'])

reexp = re.compile(r"(?P<version>[\d.dev]+[\d]+)[+]?(g(?P<gittag>\w+))?[.]?(d(?P<dirtydate>[\d]+))?")

ver = reexp.match(vers.version)


class TagVersion(MarxsElement):
    def __init__(self, **kwargs):
        self.origin = kwargs.pop('origin', 'unknown')
        self.creator = kwargs.pop('creator', 'MARXS')

    def __call__(self, photons, *args, **kwargs):
        photons.meta['ARCUSVER'] = (ver.group('version'), 'ARCUS code version')
        if not ver.group('gittag') is None:
            photons.meta['ARCUSGIT'] = (ver.group('gittag'),
                                        'Git hash of ARCUS code')
        if not ver.group('dirtydate') is None:
            photons.meta['ARCUSTIM'] = (ver.group('dirtydate'),
                                        'Date of dirty version ARCUS code')
        photons.meta['ARCDATHA'] = (git_hash.decode()[:10],
                                    'Git hash of simulation input data')
        photons.meta['ARCDATDA'] = (git_info.decode()[:19],
                                    'Commit time of simulation input data')
        photons.meta['ORIGIN'] = (self.origin,
                                  'Institution where file was created')
        photons.meta['CREATOR'] = (self.creator,
                                   'Person or program creating the file')
        photons.meta['DATE'] = datetime.now().isoformat()[:10]
        photons.meta['SATELLIT'] = 'ARCUS'
        photons.meta['TELESCOP'] = ('ARCUS', 'placeholder - no name registered with OGIP')
        photons.meta['INSTRUME'] = ('ARCUSCAM',  'placeholder - no name registered with OGIP')
        photons.meta['FILTER'] = ('NONE', 'filter information')
        photons.meta['GRATING'] = ('ARCUSCAT',  'placeholder - no name registered with OGIP')

        return photons


tagversion = TagVersion()
