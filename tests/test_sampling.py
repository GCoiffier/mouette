import pytest
import numpy as np
import mouette as M
from mouette import geometry as geom
from mouette import sampling
from data import *

@pytest.mark.parametrize("bb",[
    geom.AABB((0.,0.), (1.,1.)),
    geom.AABB((-1, 0.5), (0., 1.3)),
    geom.AABB.unit_cube(3),
    geom.AABB((-1,-1,-1),(1.,1.,1.)),
    geom.AABB((2.,0.,0.),(3.,3.,1.)),
    geom.AABB((0.,1.,0.,1.,0.,1.), (2.,2.,2.,2.,2.,2.))
])
def test_sample_AABB(bb):
    pts = sampling.sample_AABB(bb, 100)
    assert pts.shape == (100,bb.dim)
    assert np.all( bb.mini <= np.min(pts, axis=0))
    assert np.all( bb.maxi >= np.max(pts, axis=0))

@pytest.mark.parametrize("bb",[
    geom.AABB((0.,0.), (1.,1.)),
    geom.AABB((2.,0.,0.),(3.,3.,1.)),
])
def test_sample_AABB_pointcloud(bb):
    pc = sampling.sample_AABB(bb,20,return_point_cloud=True)
    assert isinstance(pc, M.mesh.PointCloud)
    assert len(pc.vertices)==20

def test_sample_AABB_pointcloud_fail():
    bb = geom.AABB.unit_cube(7)
    try:
        _ = sampling.sample_AABB(bb,20,return_point_cloud=True)
        assert False
    except ValueError:
        assert True

def test_sample_ball():
    center = M.Vec.random(3)
    pts = sampling.sample_ball(center, 1., 100)
    assert pts.shape == (100,3)
    assert np.all(np.linalg.norm(pts-center, axis=1) <= 1.)
    pc = sampling.sample_ball(M.Vec(0,0,1), 1., 100, return_point_cloud=True)
    assert len(pc.vertices)==100
    
def test_sample_sphere():
    center = M.Vec.random(3)
    pts = sampling.sample_sphere(center, 1., 100)
    assert pts.shape == (100,3)
    assert np.all( abs(np.linalg.norm(pts-center, axis=1) - 1.) < 1e-10)
    pc = sampling.sample_sphere(center, 1., 100, return_point_cloud=True)
    assert len(pc.vertices)==100

@pytest.mark.parametrize("m", [surf_circle(), surf_pointy()])
def test_sample_polyline(m):
    bnd = M.processing.extract_boundary_of_surface(m)[0]
    pts = sampling.sample_polyline(bnd, 100)
    assert pts.shape == (100,3)
    pc = sampling.sample_polyline(bnd, 100, return_point_cloud=True)
    assert isinstance(pc, M.mesh.PointCloud)
    assert len(pc.vertices)==100

    
@pytest.mark.parametrize("m", [surf_spline(), surf_half_sphere()])
def test_sample_surface(m):
    pts = sampling.sample_surface(m, 100)
    assert pts.shape == (100,3)
    pc = sampling.sample_surface(m, 100, return_point_cloud=True, return_normals=True)
    assert isinstance(pc, M.mesh.PointCloud)
    assert len(pc.vertices)==100
    assert pc.vertices.has_attribute("normals")
