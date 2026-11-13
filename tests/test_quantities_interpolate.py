import mouette as M
from data import *
import numpy as np

@pytest.mark.parametrize("weight", ["uniform", "area", "angle"])
@pytest.mark.parametrize("persistent_in", [True, False])
@pytest.mark.parametrize("persistent_out", [True, False])
@pytest.mark.parametrize("dim", [1,3])
def test_vertex_to_face_to_vertex(weight, persistent_in, persistent_out, dim):
    mesh = ALL_SURFACES["half_sphere"]
    if persistent_in:
        vert_attr = np.ones((len(mesh.vertices), dim), dtype=float)
    else:
        vert_attr = mesh.vertices.create_attribute("test_in", float, dim, sparse=False, default_value=1.)
    face_attr = M.attributes.interpolate_vertices_to_faces(mesh, vert_attr, "face_attr", persistent=persistent_out)
    vert_attr2 = M.attributes.interpolate_faces_to_vertices(mesh, face_attr, "test_out", weight=weight, persistent=persistent_out)
    assert np.allclose(vert_attr, vert_attr2)

@pytest.mark.parametrize("weight", ["uniform", "angle"])
@pytest.mark.parametrize("persistent_in", [True, False])
@pytest.mark.parametrize("persistent_out", [True, False])
@pytest.mark.parametrize("dim", [1,3])
def test_vertex_to_corner_to_vertex(weight, persistent_in, persistent_out, dim):
    mesh = ALL_SURFACES["half_sphere"]
    if persistent_in:
        vert_attr = np.ones((len(mesh.vertices), dim), dtype=float)
    else:
        vert_attr = mesh.vertices.create_attribute("test_in", float, dim, sparse=False, default_value=1.)
    corner_attr = M.attributes.scatter_vertices_to_corners(mesh, vert_attr, "corner_attr", persistent=persistent_out)
    vert_attr2 = M.attributes.interpolate_corners_to_vertices(mesh, corner_attr, "vert_attr2", weight=weight, persistent=persistent_out)
    assert np.allclose(vert_attr, vert_attr2)


@pytest.mark.parametrize("weight", ["uniform", "angle"])
@pytest.mark.parametrize("persistent_in", [True, False])
@pytest.mark.parametrize("persistent_out", [True, False])
@pytest.mark.parametrize("dim", [1,3])
def test_face_to_corner_to_face(weight, persistent_in, persistent_out, dim):
    mesh : M.mesh.SurfaceMesh = ALL_SURFACES["half_sphere"]
    if persistent_in:
        face_attr = np.ones((len(mesh.faces), dim), dtype=float)
    else:
        face_attr = mesh.faces.create_attribute("test_in", float, dim, sparse=False, default_value=1.)
    corner_attr = mesh.face_corners.create_attribute("test", float)
    corner_attr = M.attributes.scatter_faces_to_corners(mesh, face_attr, "corner_attr", persistent=persistent_out)
    face_attr2 = M.attributes.interpolate_corners_to_faces(mesh, corner_attr, "face_attr2", weight=weight, persistent=persistent_out)
    assert np.allclose(face_attr, face_attr2)
