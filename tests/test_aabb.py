import mouette as M
from mouette import geometry as geom
import numpy as np

def test_init():
    bb = geom.AABB((0.,0.,0.), (1.,1.,1.))
    assert np.all(bb.mini == np.array((0.,0.,0.)))
    assert np.all(bb.maxi == np.array((1.,1.,1.)))

def test_init2():
    bb1 = geom.AABB((0.,0., 0., 0.), (1.,1.,1., 1.))
    bb2 = geom.AABB.unit_cube(4)
    assert np.all(bb1.mini == bb2.mini)
    assert np.all(bb1.maxi == bb2.maxi)


def test_dim():
    bb = geom.AABB((-1.,-1.,-1), (1.,1.,1.))
    assert np.all(bb.span == np.array((2.,2.,2.)))
    assert bb.center.x == 0.
    assert bb.center.y == 0.
    assert bb.center.z == 0.

def test_contains():
    for _ in range(10):
        pt_max = 1.001 + geom.Vec.random(3)
        bb = geom.AABB(geom.Vec(-.01, -.01, -.01), pt_max)
        query_pt = geom.Vec.random(3)
        assert bb.contains_point(query_pt)
        assert not bb.contains_point(query_pt +geom.Vec(2.,0.,0.))

def test_do_intersect():
    b1 = geom.AABB((0.,0.,0.), (1.,1.,1.))
    b2 = geom.AABB((0.9,0.9,0.9), (1., 1.2, 1.))
    b3 = geom.AABB((1.1, 1.1, 1.1), (2.2, 2.2, 2.2))
    
    assert geom.AABB.do_intersect(b1,b2)
    assert not geom.AABB.do_intersect(b1,b3)

def test_intersection1():
    b1 = geom.AABB((0., 0., 0.), (100., 100., 100.))
    b2 = geom.AABB((-10., -10., -10), (1., 1., 1.))

    bb_int = b1 & b2
    assert np.all(bb_int.mini == geom.Vec(0.,0.,0.))
    assert np.all(bb_int.maxi == geom.Vec(1.,1.,1.))

def test_intersection2():
    b1 = geom.AABB((0., 0., 0.), (1., 1., 1.))
    b2 = geom.AABB((2., 2., 2.), (3., 3., 3.))
    bb_int = b1 & b2
    assert bb_int.is_empty()

def test_union1():
    # b1 and b2 disjoint
    b1 = geom.AABB((0., 0., 0.), (8., 1., 1.))
    b2 = geom.AABB((10., 10., 10.), (10., 80., 10.))

    bb_un = b1 | b2
    assert np.all(bb_un.mini == geom.Vec(0.,0.,0.))
    assert np.all(bb_un.maxi == geom.Vec(10.,80.,10.))
    
def test_union2():
    # b2 inside b1
    b1 = geom.AABB((0., 0., 0.),  (10., 10., 10.))
    b2 = geom.AABB((1., 1., 1), (2., 2., 2.)) 

    bb_un = b1 | b2
    assert np.all(bb_un.mini == b1.mini)
    assert np.all(bb_un.maxi == b1.maxi)

def test_invalid_dim():
    bb = geom.AABB.unit_cube(5,centered=True)
    assert bb.dim == 5
    pt = M.Vec.random(4)
    try:
        _ = bb.project(pt)
        assert False
    except geom.AABB.IncompatibleDimensionError as e:
        assert True

def test_projection():
    box = geom.AABB.unit_cube(8)
    samples = M.sampling.sample_AABB(box, 10_000, "uniform")
    for _ in range(10):
        pos = M.Vec.random(8)*10
        proj = box.project(pos)
        d_proj = geom.distance(pos,proj)
        dist = [geom.distance(pos, sample) for sample in samples]
        assert all([d >= d_proj for d in dist])