"""Mouette

Maillages, Outils Et Traitement auTomatique de la géométriE
(Meshes, Tools and Geometry Processing)
"""

from . import config
from . import utils
from . import mesh
from . import attributes
from . import procedural
from . import operators
from . import geometry
from . import processing
from . import optimize
from . import sampling
from . import spatial
from . import splines

from .processing import framefield
from .processing import parametrization

from .geometry.vector import Vec
from .geometry import transform
from .utils import Logger
from .processing.worker import Worker
from .mesh.mesh_attributes import ArrayAttribute, Attribute
