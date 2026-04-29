import argparse
import numpy as np
import mouette as M
from scipy.spatial import KDTree


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="input file path. A point cloud representation (like a .xyz file) is expected")
    args = parser.parse_args()

    input_pc = M.mesh.load(args.input_file)
    assert isinstance(input_pc, M.mesh.PointCloud)
    points = np.asarray(input_pc.vertices)

    kdtree = KDTree(points)
    pcn = M.processing.PointCloudNormalEstimator(compute_curvature=True)
    pcn.run(input_pc, kdtree=kdtree)

    M.mesh.save(input_pc, "output.geogram_ascii")
    _,nn = kdtree.query(points, 2)
    vecs = points[nn[:,0], :] - points[nn[:,1], :]
    scale = 5.*np.mean(np.linalg.norm(vecs, axis=1))
    M.mesh.save(pcn.normals_as_vector_field(scale), "output_normals.mesh")

