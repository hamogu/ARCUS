from datetime import datetime

from marxs.base import MarxsElement
from . import version
from .load_csv import get_git_hash, string_git_info


class TagVersion(MarxsElement):
    def __init__(self, **kwargs):
        self.origin = kwargs.pop('origin', 'unknown')
        self.creator = kwargs.pop('creator', 'MARXS')

    def __call__(self, photons, *args, **kwargs):
        photons.meta['ARCUSVER'] = (version.version, 'ARCUS code version')
        photons.meta['ARCUSGIT'] = (version.githash, 'Git hash of ARCUS code')
        photons.meta['ARCUSTIM'] = (version.timestamp, 'Commit time')
        photons.meta['ARCDATHA'] = (get_git_hash()[:10], 'Git hash of simulation input data')
        photons.meta['ARCDATGI'] = (string_git_info()[:20], '')
        photons.meta['ORIGIN'] = (self.origin, 'Institution where file was created')
        photons.meta['CREATOR'] = (self.creator, 'Person or program creating the file')
        photons.meta['DATE'] = datetime.now().isoformat()[:10]
        photons.meta['SATELLIT'] = 'ARCUS'

        return photons

tagversion = TagVersion()
