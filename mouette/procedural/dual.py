from ..attributes import face_barycenter, face_circumcenter
from ..mesh.datatypes import *
from ..mesh.mesh import from_arrays

@allowed_mesh_types(SurfaceMesh)
def dual_mesh(mesh: SurfaceMesh, mode:str = "barycenter") -> SurfaceMesh:
    """Computes the dual mesh of a mesh

    Args:
        mesh (SurfaceMesh): input surface
        mode (str): which position

    Returns:
        SurfaceMesh
    """
    if mode.lower() == "barycenter":
        dual_pts = face_barycenter(mesh, persistent=False)
    elif mode.lower() == "circumcenter":
        dual_pts = face_circumcenter(mesh, persistent=False)

    raw_dual = RawMeshData()
    for F in mesh.id_faces:
        raw_dual.vertices.append(dual_pts[F])
    for V in mesh.id_vertices:
        raw_dual.faces.append(mesh.connectivity.vertex_to_faces(V))
    return raw_dual.instantiate()