from marxs.base import MarxsElement
from . import version
from load_csv import get_git_hash, string_git_info


class TagVersion(MarxsElement):
    def __call__(self, photons, *args, **kwargs):
        photons.meta['ARCUSVER'] = (version.version, 'ARCUS code version')
        photons.meta['ARCUSGIT'] = (version.githash, 'Git hash of ARCUS code')
        photons.meta['ARCUSTIM'] = (version.timestamp, 'Commit time')
        photons.meta['ARCDATHA'] = (get_git_hash()[:10], 'Git hash of simulation input data')
        photons.meta['ARCDATGI'] = (string_git_info()[:20], '')
        return photons

tagversion = TagVersion()
