import mouette as M
from mouette.processing import parametrization
import sys

if len(sys.argv)<2:
    print("Usage: `python orbifold_tutte.py <path/to/mesh>")

mesh = M.mesh.load(sys.argv[1])
orbTutte = parametrization.OrbifoldTutteEmbedding(mesh,verbose=True)

M.mesh.save(mesh, "output.geogram_ascii")
M.mesh.save(orbTutte.flat_mesh, "output_flat.geogram_ascii")