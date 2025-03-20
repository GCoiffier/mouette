import pytest
import mouette as M
from mouette.processing import PointCloudNormalEstimator
from mouette.sampling import sample_surface
from data import *


@pytest.mark.parametrize("m",[surf_half_sphere()])
def test_normal_estimation(m):
    points, true_normals = sample_surface(m, 10_000, return_normals=True)
    pc = M.mesh.from_arrays(points)
    pcn = PointCloudNormalEstimator(n_neighbors=10)
    pcn.run(pc)
    estimated_normals = pcn.normals.as_array(len(pc.vertices))
    dot_products = [np.dot(aa,bb)>0 for aa, bb in zip(true_normals,estimated_normals)]
    assert all(dot_products)


@pytest.mark.parametrize("m", [surf_half_sphere(), surf_cube()])
def test_normal_estimation_no_save_on_mesh(m):
    pc = sample_surface(m, 2000, return_point_cloud=True)
    pcn = PointCloudNormalEstimator(save_on_pc=False)
    pcn.run(pc)
    assert not pc.vertices.has_attribute("normals")
    assert pcn.normals.shape == (2000, 3)

