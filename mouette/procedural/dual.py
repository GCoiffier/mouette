from ..attributes import face_barycenter, face_circumcenter
from ..mesh.datatypes import *
from ..mesh.mesh_data import RawMeshData
from ..mesh.mesh import _instanciate_raw_mesh_data

@allowed_mesh_types(SurfaceMesh)
def dual_mesh(mesh: SurfaceMesh, mode:str = "barycenter") -> SurfaceMesh:
    """Computes the dual mesh of a mesh

    Args:
        mesh (SurfaceMesh): input surface
        mode (str): which position

    Returns:
        SurfaceMesh
    """
    out = RawMeshData()
    if mode.lower() == "barycenter":
        dual_pts = face_barycenter(mesh, persistent=False)
    elif mode.lower() == "circumcenter":
        dual_pts = face_circumcenter(mesh)
    for F in mesh.id_faces:
        out.vertices.append(dual_pts[F])
    for V in mesh.id_vertices:
        out.faces.append(mesh.connectivity.vertex_to_faces(V))
    return _instanciate_raw_mesh_data(out, 2)