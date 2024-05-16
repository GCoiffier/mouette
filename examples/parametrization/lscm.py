"""
Mouette example files.
Parametrization using LSCM
"""

import mouette as M
from mouette.processing import parametrization
import sys

if len(sys.argv)<2:
    print("Usage: `python lscm.py <path/to/mesh>")

mesh = M.mesh.load(sys.argv[1])
lscm = parametrization.LSCM(mesh, verbose=True)()

M.mesh.save(mesh, "output.geogram_ascii")
M.mesh.save(lscm.flat_mesh, "output_flat.geogram_ascii")