import pygeomesh as GEO
import pytest
import os
import numpy as np

### Polylines ###

def pl_one_edge():
    m = GEO.mesh.new_polyline()
    m.vertices += [GEO.Vec(0.,0.,0.), GEO.Vec(1.,0.,0.)]
    m.edges.append((0,1))
    return m

def pl_sample1():
    m = GEO.mesh.new_polyline()
    m.vertices += [
        GEO.Vec(0.,0.,0.),
        GEO.Vec(1.,0.,1.),
        GEO.Vec(2.,1.,0.)
    ]
    m.edges += [(0,1), (1,2)]
    return m

def pl_sample2():
    m = GEO.mesh.new_polyline()
    m.vertices += [
        GEO.Vec.random(3) for _ in range(6)
    ]
    m.edges += [(0,1), (1,5), (3,5), (2,3), (0,2), (0,3), (3,4), (2,4)]
    return m

def pl_two_components():
    m = GEO.mesh.new_polyline()
    m.vertices += [
        GEO.Vec.random(3) for _ in range(10)
    ]
    m.edges += [(0,1), (1,4), (3,5), (2,3), (0,2), (0,3), (2,4), (6,7), (7,8), (7,9), (6,9)]

    return m

polylines = [
    pl_one_edge(),
    pl_sample1(),
    pl_sample2(),
    pl_two_components()
]