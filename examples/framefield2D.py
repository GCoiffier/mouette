"""
Mouette example files.
Computation of a 2D frame field on vertices
"""

import mouette as M
from mouette.processing import framefield
import sys

if len(sys.argv)<2:
    print("Usage: `python framefield2D.py <path/to/mesh> <order>")

order = int(sys.argv[2]) if len(sys.argv)>3 else 4

mesh = M.mesh.load(sys.argv[1])
ff = framefield.FrameField2DVertices(mesh, order=order, feature_edges=True, verbose=True)
ff.run()
ff.flag_singularities()

M.mesh.save(mesh, "output_model.geogram_ascii")
M.mesh.save(ff.export_as_mesh(), "output_FF.geogram_ascii")

