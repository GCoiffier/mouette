"""
M.processing
submodule for algorithms applied on meshes
"""

from .misc import *
from .cutting import SingularityCutter
from .features import FeatureEdgeDetector
from .expmap import DiscreteExponentialMap

from .paths import shortest_path, shortest_path_to_border, shortest_path_to_vertex_set
from .border import extract_border_cycle, extract_border_cycle_all, extract_curve_boundary, extract_surface_boundary

from . import trees
from .parametrization import distortion
from .connection import SurfaceConnectionFaces, SurfaceConnectionVertices, FlatConnectionVertices, FlatConnectionFaces