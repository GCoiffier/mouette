from mouette import geometry as geom
import numpy as np

###### BB2D ######

def test_init():
    bb = geom.BB2D(0.,0.,1.,1.)
    assert bb.left == 0.
    assert bb.right == 1.
    assert bb.bottom == 0.
    assert bb.top == 1.

def test_dim():
    bb = geom.BB2D(-1.,-1., 1., 1.)
    assert bb.width == 2.
    assert bb.height == 2.
    assert bb.center.x == 0.
    assert bb.center.y == 0.

def test_contains():
    for _ in range(10):
        pt_max = 1.001 + geom.Vec.random(2)
        bb = geom.BB2D(-0.01, -0.01, pt_max[0], pt_max[1])
        query_pt = geom.Vec.random(2)
        assert bb.contains_point(query_pt)
        assert not bb.contains_point(query_pt + np.array((2.,0.)))


def test_do_intersect():
    b1 = geom.BB2D(0.,0.,1.,1.)
    b2 = geom.BB2D(0.9,0.9,1.,1.2)
    b3 = geom.BB2D(1.1,1.1,2.2,2.2)
    
    assert geom.BB2D.do_intersect(b1,b2)
    assert not geom.BB2D.do_intersect(b1,b3)

def test_intersection1():
    b1 = geom.BB2D(0., 0.,100.,100.)
    b2 = geom.BB2D(-10., -10., 1.,1.)

    bb_int = b1 & b2
    assert bb_int.left == 0.
    assert bb_int.right == 1.
    assert bb_int.bottom == 0.
    assert bb_int.top == 1. 

def test_intersection2():
    b1 = geom.BB2D(0., 0., 1., 1.)
    b2 = geom.BB2D(2., 2., 3., 3.)

    bb_int = b1 & b2
    assert bb_int is None

def test_union1():
    # b1 and b2 disjoint
    b1 = geom.BB2D(0., 0., 8.,1.)
    b2 = geom.BB2D(10., 10., 10., 80.)

    bb_un = b1 | b2
    assert bb_un.left == 0.
    assert bb_un.right == 10.
    assert bb_un.bottom == 0.
    assert bb_un.top == 80. 

def test_union2():
    # b2 inside b1
    b1 = geom.BB2D(0., 0., 10.,10.)
    b2 = geom.BB2D(1., 1., 2., 2.) 

    bb_un = b1 | b2
    assert bb_un.left == 0.
    assert bb_un.right == 10.
    assert bb_un.bottom == 0.
    assert bb_un.top == 10. 

###### BB3D ######

def test_init():
    bb = geom.BB3D(0.,0.,0., 1.,1.,1.)
    assert np.all(bb.min_coords == np.array((0.,0.,0.)))
    assert np.all(bb.max_coords == np.array((1.,1.,1.)))

def test_dim():
    bb = geom.BB3D(-1.,-1.,-1, 1.,1.,1.)
    assert np.all(bb.span == np.array((2.,2.,2.)))
    assert bb.center.x == 0.
    assert bb.center.y == 0.
    assert bb.center.z == 0.

def test_contains():
    for _ in range(10):
        pt_max = 1.001 + geom.Vec.random(3)
        bb = geom.BB3D(geom.Vec(-.01, -.01, -.01), pt_max)
        query_pt = geom.Vec.random(3)
        assert bb.contains_point(query_pt)
        assert not bb.contains_point(query_pt +geom.Vec(2.,0.,0.))

def test_do_intersect():
    b1 = geom.BB3D(0.,0.,0., 1.,1.,1.)
    b2 = geom.BB3D(0.9,0.9,0.9, 1., 1.2, 1.)
    b3 = geom.BB3D(1.1, 1.1, 1.1, 2.2, 2.2, 2.2)
    
    assert geom.BB3D.do_intersect(b1,b2)
    assert not geom.BB3D.do_intersect(b1,b3)

def test_intersection1():
    b1 = geom.BB3D(0., 0., 0., 100., 100., 100.)
    b2 = geom.BB3D(-10., -10., -10, 1., 1., 1.)

    bb_int = b1 & b2
    assert np.all(bb_int.min_coords == geom.Vec(0.,0.,0.))
    assert np.all(bb_int.max_coords == geom.Vec(1.,1.,1.))

def test_intersection2():
    b1 = geom.BB3D(0., 0., 0., 1., 1., 1.)
    b2 = geom.BB3D(2., 2., 2., 3., 3., 3.)
    bb_int = b1 & b2
    assert bb_int is None

def test_union1():
    # b1 and b2 disjoint
    b1 = geom.BB3D(0., 0., 0., 8., 1., 1.)
    b2 = geom.BB3D(10., 10., 10., 10., 80., 10.)

    bb_un = b1 | b2
    assert np.all(bb_un.min_coords == geom.Vec(0.,0.,0.))
    assert np.all(bb_un.max_coords == geom.Vec(10.,80.,10.))
    

def test_union2():
    # b2 inside b1
    b1 = geom.BB3D(0., 0., 0.,  10., 10., 10.)
    b2 = geom.BB3D(1., 1., 1, 2., 2., 2.) 

    bb_un = b1 | b2
    assert np.all(bb_un.min_coords == b1.min_coords)
    assert np.all(bb_un.max_coords == b1.max_coords)