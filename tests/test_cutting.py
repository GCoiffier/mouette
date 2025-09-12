import mouette as M
from data import *
import pytest

@pytest.mark.parametrize("mesh", [surf_spot()])
def test_cutting_edges(mesh : M.mesh.SurfaceMesh):
    cutter = M.processing.SurfaceMeshCutter(mesh, verbose=True)
    vertex_path1 = M.processing.shortest_path(mesh, 796, [995])[995]
    vertex_path2 = M.processing.shortest_path(mesh, 96, [861])[861]
    edge_path1 = [mesh.connectivity.edge_id(*e) for e in M.utils.consecutive_pairs(vertex_path1)]
    edge_path2 = [mesh.connectivity.edge_id(*e) for e in M.utils.consecutive_pairs(vertex_path2)]
    cutter.cut(edge_path1 + edge_path2)

    # test if the output mesh is indeed a disk topology
    assert M.attributes.euler_characteristic(cutter.cut_mesh)==1 and len(cutter.cut_mesh.boundary_vertices)>0
    
    # test that all boundary vertices of the cut mesh were vertices on a cut of the original mesh
    cut_vertices = set(vertex_path1 + vertex_path2)
    for v in cutter.cut_mesh.boundary_vertices:
        assert cutter.ref_vertex(v) in cut_vertices

    # test that all corners are the same
    for c,v in enumerate(cutter.cut_mesh.face_corners):
        assert cutter.ref_vertex(v) == mesh.face_corners[c]
        assert mesh.face_corners.adj(c) == cutter.cut_mesh.face_corners.adj(c)
    
    # test generating the cutgraph
    cut_graph = cutter.cut_graph
    assert isinstance(cut_graph, M.mesh.PolyLine)
    assert len(cut_graph.edges) == len(cutter.cut_edges)


@pytest.mark.parametrize("mesh,singus", [
    (surf_pointy(), [300]),
    (surf_circle(), [155, 200, 217, 334])
])
def test_cutting_edges_to_boundary(mesh : M.mesh.SurfaceMesh, singus):
    cutter = M.processing.SurfaceMeshCutter(mesh, verbose=True)
    cut_edges = set()
    cut_vertices = set()
    for s in singus:
        vertex_path = M.processing.shortest_path_to_border(mesh,s)
        cut_vertices.update(vertex_path)
        cut_edges.update({mesh.connectivity.edge_id(*e) for e in M.utils.consecutive_pairs(vertex_path)})
    cutter.cut(cut_edges)

    assert M.attributes.euler_characteristic(cutter.cut_mesh)==1
    assert len(cutter.cut_mesh.boundary_vertices)>len(mesh.boundary_vertices)
    
    # test that all boundary vertices of the cut mesh were vertices on a cut of the original mesh
    for v in cutter.cut_mesh.boundary_vertices:
        # boundary mesh is either a cut vertex or was already on the boundary
        ov = cutter.ref_vertex(v)
        assert (ov in cut_vertices) or (mesh.is_vertex_on_border(ov))


@pytest.mark.parametrize("mesh,singus,strategy",[
    (surf_spot(),[796, 978, 995],"simple"),
    (surf_spot(),[796, 978, 995],"shortest"),
    (surf_spot(),[796, 978, 995],"limited"),
    (surf_spot(),[796, 978, 995],"auto"),
    (surf_cube_subdiv(),[100, 141, 59, 68], "simple"),
    (surf_cube_subdiv(),[100, 141, 59, 68], "shortest"),
    (surf_cube_subdiv(),[100, 141, 59, 68], "limited"),
    (surf_cube_subdiv(),[100, 141, 59, 68], "features"),
    (surf_cube_subdiv(),[100, 141, 59, 68], "auto"),
    (surf_torus(),[],"shortest"),
    (surf_torus(),[],"auto"),
])
def test_singularity_cutting(mesh : M.mesh.SurfaceMesh, singus, strategy):
    if strategy=="features":
        feat = M.processing.FeatureEdgeDetector(compute_feature_graph=False, flag_corners=False)(mesh)
    else:
        feat = None
    cutter = M.processing.SingularityCutter(mesh, singus, strategy, features=feat, verbose=True)
    cutter.cut()
    
    assert M.attributes.euler_characteristic(cutter.cut_mesh)==1
    for v in singus:
        assert all([cutter.cut_mesh.is_vertex_on_border(dv) for dv in cutter.duplicated_vertices(v)])
    