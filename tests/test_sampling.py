import pytest
from mouette import geometry as geom
import numpy as np
from mouette.processing import sampling
from data import *

@pytest.mark.parametrize("bb",[
    geom.BB2D(0.,0.,1.,1.),
    geom.BB2D(-1,-1,1.,1.),
    geom.BB2D(2.,0.,3.,3.)
])
def test_sample_BB2D(bb):
    pts = sampling.sample_bounding_box_2D(bb, 100)
    assert pts.shape == (100,2)
    pmin = np.min(pts,axis=0)
    pmax = np.max(pts,axis=0)
    assert bb.left <= pmin[0] <= pmax[0] <= bb.right
    assert bb.bottom <= pmin[1] <= pmax[1] <= bb.top
    pc = sampling.sample_bounding_box_2D(bb,20,return_point_cloud=True)
    assert len(pc.vertices)==20


@pytest.mark.parametrize("bb",[
    geom.BB3D(0.,0.,0.,1.,1.,1.),
    geom.BB3D(-1,-1,-1,1.,1.,1.),
    geom.BB3D(2.,0.,0.,3.,3.,1.)
])
def test_sample_BB3D(bb):
    pts = sampling.sample_bounding_box_3D(bb, 100)
    assert pts.shape == (100,3)
    assert np.all( bb.min_coords <= np.min(pts, axis=0))
    assert np.all( bb.max_coords >= np.max(pts, axis=0))
    pc = sampling.sample_bounding_box_3D(bb,20,return_point_cloud=True)
    assert len(pc.vertices)==20


@pytest.mark.parametrize("m", [surf_circle(), surf_pointy()])
def test_sample_polyline(m):
    bnd = M.processing.extract_curve_boundary(m)[0]
    pts = sampling.sample_points_from_polyline(bnd, 100)
    assert pts.shape == (100,3)
    pc = sampling.sample_points_from_polyline(bnd, 100, return_point_cloud=True)
    assert isinstance(pc, M.mesh.PointCloud)
    assert len(pc.vertices)==100

    
@pytest.mark.parametrize("m", [surf_spline(), surf_half_sphere()])
def test_sample_surface(m):
    pts = sampling.sample_points_from_surface(m, 100)
    assert pts.shape == (100,3)
    pc = sampling.sample_points_from_surface(m, 100, return_point_cloud=True, return_normals=True)
    assert isinstance(pc, M.mesh.PointCloud)
    assert len(pc.vertices)==100
    assert pc.vertices.has_attribute("normals")
