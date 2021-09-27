from typing import Dict

from ...phi import math, struct

from ._geom import Geometry, _fill_spatial_with_singleton
from ..math import wrap

from ._sphere import Sphere

class InverseSphere(Sphere):
    """
    N-dimensional sphere.
    Defined through center position and radius.

    Args:

    Returns:
    """

    def __init__(self, center, radius):
        super().__init__(center,radius)

    def lies_inside(self, location):
        distance_squared = math.sum((location - self.center) ** 2, dim='vector')
        return distance_squared >= self.radius ** 2                          #normal <
