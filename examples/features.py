import mouette as M
import sys

try:
    file_path = sys.argv[1]
except:
    print("Usage: python features.py <path/to/surface/mesh>")


m = M.mesh.load(file_path)
if not isinstance(m, M.mesh.SurfaceMesh):
    print("provided mesh is not a surface mesh. Aborting.")
    exit()

feat = M.processing.FeatureEdgeDetector()
feat.run(m)

M.mesh.save(m,"surface.mesh")
M.mesh.save(feat._feature_graph, "features.mesh")