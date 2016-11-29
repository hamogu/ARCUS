
from copy import deepcopy
import transforms3d
import arcus

arcus_orig = arcus.arcus_extra_det
arcus_orig_m = arcus.arcus_extra_det_m

def move_element(aff, element):
    # If this is a sequence or parallel, recurse down to the leaves
    if hasattr(element, 'elements'):
        for e in element.elements:
            move_element(aff, e)
    else:
        # Some elements are global and don't have to be moved, e.g. GobalEnergyFilter
        if hasattr(element, "pos4d"):
            element.pos4d = np.dot(aff, element.pos4d)

def move_boom(aff, arcus):
    # For three elements of arcus are part of the boom: aper, mirror, gas
    for i in range(3):
        move_element(aff, arcus.elements[i])


mission = deepcopy(arcus_orig)

aff = transforms3d.affines.compose([0, 100, 0], np.eye(3), np.ones(3))
