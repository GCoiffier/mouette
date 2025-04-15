import mouette as M
from data import *
import numpy as np

@pytest.mark.parametrize("weight", ["uniform", "area", "angle"])
def test_vertex_to_face_to_vertex(weight):
    mesh = surf_half_sphere()
    vert_attr = mesh.vertices.register_array_as_attribute("test_in", np.ones(len(mesh.vertices)))
    vert_attr2 = mesh.vertices.create_attribute("test_out", float)
    face_attr = mesh.faces.create_attribute("test", float)

    M.attributes.interpolate_vertices_to_faces(mesh, vert_attr, face_attr)
    M.attributes.interpolate_faces_to_vertices(mesh, face_attr, vert_attr2, weight=weight)
    assert np.all(np.isclose(vert_attr.as_array(len(mesh.vertices)), vert_attr2.as_array(len(mesh.vertices))))


@pytest.mark.parametrize("weight", ["uniform", "angle"])
def test_vertex_to_corner_to_vertex(weight):
    mesh = surf_half_sphere()
    vert_attr = mesh.vertices.register_array_as_attribute("test_in", np.ones(len(mesh.vertices)))
    vert_attr2 = mesh.vertices.create_attribute("test_out", float)
    corner_attr = mesh.face_corners.create_attribute("test", float)

    M.attributes.scatter_vertices_to_corners(mesh, vert_attr, corner_attr)
    M.attributes.average_corners_to_vertices(mesh, corner_attr, vert_attr2, weight=weight)
    assert np.all(np.isclose(vert_attr.as_array(len(mesh.vertices)), vert_attr2.as_array(len(mesh.vertices))))


@pytest.mark.parametrize("weight", ["uniform", "angle"])
def test_face_to_corner_to_face(weight):
    mesh = surf_half_sphere()
    face_attr = mesh.faces.register_array_as_attribute("test_in", np.ones(len(mesh.faces)))
    face_attr2 = mesh.faces.create_attribute("test_out", float)
    corner_attr = mesh.face_corners.create_attribute("test", float)

    M.attributes.scatter_faces_to_corners(mesh, face_attr, corner_attr)
    M.attributes.average_corners_to_faces(mesh, corner_attr, face_attr2, weight=weight)
    assert np.all(np.isclose(face_attr.as_array(len(mesh.faces)), face_attr2.as_array(len(mesh.faces))))