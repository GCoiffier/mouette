import pytest
import mouette as M
from data import *
from utils import compare_mesh



@pytest.mark.parametrize("pl", POLYLINES.items())
def test_split_polyline_edges(pl):
    name, pl = pl
    split = M.mesh.subdivision.split_edges(pl)
    assert len(split.edges) == 2*len(pl.edges)
    assert len(split.vertices) == len(pl.edges) + len(pl.vertices)


@pytest.mark.parametrize("pl", POLYLINES.items())
def test_split_polyline_edges_edge_ind(pl):
    name, pl = pl
    split = M.mesh.subdivision.split_edges(pl, [0])
    assert len(split.edges) == 1+len(pl.edges)
    assert len(split.vertices) == 1+len(pl.vertices)

### Surface Subdivision ###

@pytest.mark.parametrize("mesh", TRI_SURFACES.items())
def test_loop_subdivision(mesh):
    name,mesh = mesh
    subdiv = M.mesh.subdivision.loop_subdivision(mesh)
    assert subdiv.is_triangular()
    assert len(subdiv.faces) == 4*len(mesh.faces)     


@pytest.mark.parametrize("mesh", TRI_SURFACES.items())
def test_triangulate_tri(mesh):
    name,mesh = mesh
    trimesh = M.mesh.subdivision.triangulate(mesh)
    assert trimesh.is_triangular()
    compare_mesh(trimesh, mesh)
    
   
@pytest.mark.parametrize("mesh", QUAD_SURFACES.items())
def test_triangulate_quad(mesh):      
    name,mesh = mesh
    trimesh = M.mesh.subdivision.triangulate(mesh)
    assert trimesh.is_triangular()
    assert len(trimesh.faces) == 2*len(mesh.faces)


@pytest.mark.parametrize("mesh", ALL_SURFACES.items())
def test_split_faces_as_fan(mesh):
    name,mesh = mesh
    subdiv = M.mesh.subdivision.split_faces_as_fan(mesh)
    assert subdiv.is_triangular()
    assert len(subdiv.vertices) == len(mesh.vertices) + len(mesh.faces)

@pytest.mark.parametrize("mesh", TRI_SURFACES.items())
def test_split_faces_as_fan_single_face(mesh):
    name,mesh = mesh
    subdiv = M.mesh.subdivision.split_faces_as_fan(mesh, [0])
    assert subdiv.is_triangular()
    assert len(subdiv.vertices) == len(mesh.vertices) + 1

@pytest.mark.parametrize("mesh", TRI_SURFACES.items())
def test_subdivide_triangles_3quads(mesh):
    name,mesh = mesh
    subdiv = M.mesh.subdivision.subdivide_triangles_3quads(mesh)
    assert subdiv.is_quad()
    assert len(subdiv.faces) == 3*len(mesh.faces)

@pytest.mark.parametrize("mesh", TRI_SURFACES.items())
def test_subdivide_triangles_6(mesh):
    name,mesh = mesh
    subdiv = M.mesh.subdivision.subdivide_triangles_6(mesh)
    assert subdiv.is_triangular()
    assert len(subdiv.faces)== 6*len(mesh.faces)

@pytest.mark.parametrize("mesh", TRI_SURFACES.items())
def test_split_double_boundary_edges_triangles(mesh):
    name,mesh = mesh
    mesh2 = M.mesh.subdivision.split_double_boundary_edges_triangles(mesh)
    assert True
