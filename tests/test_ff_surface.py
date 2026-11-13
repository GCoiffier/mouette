import pytest
from mouette import framefield as ff
from mouette.processing import SurfaceConnectionFaces, SurfaceConnectionVertices, SurfaceConnectionEdges, FeatureEdgeDetector
from mouette.mesh.datatypes.type_checks import BadMeshTypeException
from mouette.utils.argument_check import InvalidArgumentTypeError,InvalidArgumentValueError, InvalidRangeArgumentError
from data import *

@pytest.mark.parametrize("m",[
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"]
])
def test_connection_faces(m):
    conn = SurfaceConnectionFaces(m)
    assert True

@pytest.mark.parametrize("m",[
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"]
])
def test_connection_vertices(m):
    conn = SurfaceConnectionVertices(m)
    assert True


@pytest.mark.parametrize("m",[
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"]
])
def test_connection_edges(m):
    conn = SurfaceConnectionEdges(m)
    assert True

def test_surface_framefield_wrong_mesh():
    m = POLYLINES["one_edge"]
    try:
        _ = ff.SurfaceFrameField(m, "vertices")
        assert False
    except BadMeshTypeException as e:
        assert True

def test_surface_framefield_wrong_elements():
    m = M.mesh.RawMeshData().instantiate(2)
    try:
        _ = ff.SurfaceFrameField(m, 42)
        assert False
    except InvalidArgumentTypeError as e:
        assert True
    
    try:
        _ = ff.SurfaceFrameField(m,"foo")
        assert False
    except InvalidArgumentValueError as e:
        assert True


def test_surface_frame_field_bad_numerical_values():
    m = M.mesh.RawMeshData().instantiate(2)
    try:
        _ = ff.SurfaceFrameField(m, "vertices", n_smooth="pouet")
        assert False
    except InvalidArgumentTypeError as e:
        assert True

    try:
        _ = ff.SurfaceFrameField(m,"faces",n_smooth=-1)
        assert False
    except InvalidRangeArgumentError as e:
        assert True

@pytest.mark.parametrize("m", [
    ALL_SURFACES["circle"],
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"],
    ALL_SURFACES["pointy"]
])
def test_surface_framefield_vertices(m):
    framefield = ff.SurfaceFrameField(m, "vertices")()
    framefield.flag_singularities()
    assert m.faces.has_attribute("singuls")

def test_CADFF():
    m = ALL_SURFACES["pointy"]
    framefield = ff.SurfaceFrameField(m, "vertices", cad_correction=True)()
    framefield.flag_singularities()
    assert m.faces.has_attribute("singuls")

@pytest.mark.parametrize("m", [
    ALL_SURFACES["circle"],
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"],
    ALL_SURFACES["pointy"]
])
def test_surface_framefield_faces(m):
    framefield = ff.SurfaceFrameField(m, "faces")()
    framefield.flag_singularities()
    assert m.vertices.has_attribute("singuls")

@pytest.mark.parametrize("m", [
    ALL_SURFACES["circle"],
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"],
    ALL_SURFACES["pointy"]
])
def test_CurvatureVertices(m):
    framefield = ff.PrincipalDirections(m, "vertices")()
    framefield.flag_singularities()
    assert m.faces.has_attribute("singuls")

@pytest.mark.parametrize("m", [
    ALL_SURFACES["circle"],
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"],
    ALL_SURFACES["pointy"]
])
def test_CurvatureFaces(m):
    framefield = ff.PrincipalDirections(m, "faces")()
    framefield.flag_singularities()
    assert m.vertices.has_attribute("singuls")

@pytest.mark.parametrize("m", [
    ALL_SURFACES["circle"],
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"],
    ALL_SURFACES["pointy"]
])
def test_surface_framefield_custom_connection_vertices(m):
    conn = SurfaceConnectionVertices(m)
    _ = ff.SurfaceFrameField(m, "vertices", custom_connection=conn)()
    assert True

@pytest.mark.parametrize("m", [
    ALL_SURFACES["circle"],
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"],
    ALL_SURFACES["pointy"]
])
def test_surface_framefield_custom_connection_faces(m):
    conn = SurfaceConnectionFaces(m)
    _ = ff.SurfaceFrameField(m, "faces", custom_connection=conn)()
    assert True

def test_surface_framefield_custom_features_vertices():
    m = ALL_SURFACES["feature"]
    feat = FeatureEdgeDetector().run(m)
    _ = ff.SurfaceFrameField(m, "vertices", custom_features=feat)()
    assert True

def test_surface_framefield_custom_features_faces():
    m = ALL_SURFACES["feature"]
    feat = FeatureEdgeDetector().run(m)
    _ = ff.SurfaceFrameField(m, "faces", custom_features=feat)()
    assert True

def test_surface_framefield_custom_singus_vertices():
    m = ALL_SURFACES["circle"]
    singus = m.faces.create_attribute("singuls", float, sparse=True)
    singus[0] = 1
    singus[10] = 1
    singus[45] = 1
    singus[76] = 1
    _ = ff.SurfaceFrameField(m, "vertices", singularity_indices=singus)()
    assert True


def test_surface_framefield_custom_singus_faces():
    m = ALL_SURFACES["circle"]
    singus = m.vertices.create_attribute("singuls", float, sparse=True)
    singus[64] = 1
    singus[80] = 1
    singus[142] = 1
    singus[150] = 1
    _ = ff.SurfaceFrameField(m, "faces", singularity_indices=singus)()
    assert True

def test_surface_framefield_custom_singus_faces_invalid_topo():
    m = ALL_SURFACES["circle"]
    singus = m.vertices.create_attribute("singuls", float, sparse=True)
    singus[64] = 1
    singus[80] = -1 # singus do not verify Poincarré-Hopf
    try:
        _ = ff.SurfaceFrameField(m, "faces", singularity_indices=singus)()
        assert False
    except Exception as e:
        assert True


@pytest.mark.parametrize("m", [
    ALL_SURFACES["circle"],
    ALL_SURFACES["spline"],
    ALL_SURFACES["half_sphere"],
    ALL_SURFACES["pointy"]
])
def test_surface_framefield_edges(m):
    framefield = ff.SurfaceFrameField(m, "edges")()
    assert True