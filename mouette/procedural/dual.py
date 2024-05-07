from ..attributes import face_barycenter
from ..mesh.datatypes import *
from ..mesh.mesh_data import RawMeshData
from ..mesh.mesh import _instanciate_raw_mesh_data

@allowed_mesh_types(SurfaceMesh)
def dual_mesh(mesh: SurfaceMesh) -> SurfaceMesh:
    """Computes the dual mesh of a mesh

    Args:
        mesh (SurfaceMesh): input surface

    Returns:
        SurfaceMesh
    """
    out = RawMeshData()
    bary = face_barycenter(mesh, persistent=False)
    for F in mesh.id_faces:
        out.vertices.append(bary[F])
    for V in mesh.id_vertices:
        out.faces.append(mesh.connectivity.vertex_to_faces(V))
    return _instanciate_raw_mesh_data(out, 2)