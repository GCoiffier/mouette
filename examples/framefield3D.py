import mouette as M
from mouette.processing import framefield
import sys

if len(sys.argv)<2:
    print("Usage: `python framefield3D.py <path/to/mesh>")

m = M.mesh.load(sys.argv[1])
ff = framefield.FrameField3DCells(m, verbose=True)
ff.run() # Initializes values 
ff.flag_singularities() # Computes the singularity graph

M.mesh.save(m, "output.geogram_ascii")
M.mesh.save(ff.export_as_mesh(), "output_FF.geogram_ascii")