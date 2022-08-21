import mouette as M
import numpy as np

def compare_container(cont1, cont2):
    if len(cont1) != len(cont2): 
        return False
    for i in range(len(cont1)):
        if isinstance(cont1[i], np.ndarray):
            if (cont1[i] != cont2[i]).all():
                return False
    return True

def compare_mesh(mesh1, mesh2):
    if type(mesh1) != type(mesh2) : return False
    comp = compare_container(mesh1.vertices, mesh2.vertices)
    if hasattr(mesh1, "edges"):
        comp = comp and compare_container(mesh1.edges, mesh2.edges)
    if hasattr(mesh1, "faces"):
        comp = comp and compare_container(mesh1.faces, mesh2.faces)
    if hasattr(mesh1, "cells"):
        comp = comp and compare_container(mesh1.cells, mesh2.cells)
    return comp

def build_test_io(mesh, tmp_path, fmt, dim=None):
    filepath = tmp_path / f"test_mesh.{fmt}"
    M.mesh.save(mesh, str(filepath))
    mesh2 = M.mesh.load(str(filepath), dim)
    return compare_mesh(mesh, mesh2)