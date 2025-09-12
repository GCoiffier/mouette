"""Mouette

Maillages, Outils Et Traitement auTomatique de la géométriE
(Meshes, Tools and Geometry Processing)
"""
__version__ = "1.2.7"

from . import config
from . import utils
from . import mesh
from . import attributes
from . import procedural
from . import operators
from . import sampling
from . import spatial
from . import splines
from . import optimize

from . import processing
from .processing import framefield
from .processing import parametrization

from . import geometry
from .geometry import transform
from .geometry.vector import Vec

from .processing.worker import Worker 
from .utils.utilities import Logger