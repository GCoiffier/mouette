"""Pygeomesh
    
A simple library for handling meshes in python    
"""

from . import config
from . import utils
from . import mesh
from . import attributes
from . import procedural
from . import operators
from . import geometry
from . import processing

from .processing import framefield

from .geometry.vector import Vec
from .geometry import transform
from .utils import Logger
from .processing.worker import Worker
from .mesh.mesh_attributes import ArrayAttribute, Attribute
