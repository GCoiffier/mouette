import numpy as np
import scipy.sparse as sp
import cmath

from ..geometry import *
from ..mesh.datatypes import *
from ..attributes.misc_faces import face_area
from ..attributes.misc_cells import cell_volume
from ..attributes.misc_corners import cotangent

##### For Surface, on vertices #####

@forbidden_mesh_types(PointCloud)
def graph_laplacian(mesh : Mesh) -> sp.csc_matrix:
    """Simplest Laplacian defined on a graph. uses uniform weights for connectivity.

    Parameters:
        mesh (Mesh): the input mesh

    Returns:
        (scipy.sparse.coo_matrix) : the laplacian operator as a sparse matrix
    """
    n = len(mesh.vertices)
    m = len(mesh.edges)
    ncoeffs = 2*m+n
    data, rows, cols = np.zeros(ncoeffs),np.zeros(ncoeffs, dtype=int),np.zeros(ncoeffs, dtype=int)
    
    def add(k, i, j, x):
        data[k] = x
        rows[k] = i
        cols[k] = j
        return k+1

    k = 0
    for l in mesh.id_vertices:
        adj = mesh.connectivity.vertex_to_vertices(l)
        k = add(k,l,l,len(adj))
        for b in adj:
            k = add(k, l, b, -1)
    return sp.csc_matrix((data, (rows,cols)), shape=(n,n))


@allowed_mesh_types(SurfaceMesh)
def area_weight_matrix(mesh : SurfaceMesh, inverse:bool = False) -> sp.csc_matrix:
    """
    Returns the diagonal matrix A of area weights on vertices
    
    Laplace-beltrami operator for a 2D manifold is (A^-1)L where A is the area weight and L is the cotan matrix

    Args:
        mesh (SurfaceMesh): input mesh
        inverse (bool, optional): whether to return A or A^-1. Defaults to False.

    Returns:
        sp.csc_matrix: diagonal matrix of vertices area
    """
    A = np.zeros(len(mesh.vertices))
    area = face_area(mesh)
    for iT,T in enumerate(mesh.faces):
        for u in T:
            A[u] += area[iT] if not inverse else 1/area[iT]
    return sp.diags(A, format="csc")

@allowed_mesh_types(SurfaceMesh)
def laplacian(
    mesh : SurfaceMesh, 
    cotan : bool=True, 
    connection : "SurfaceConnectionVertices" = None, 
    order : int=4) -> sp.lil_matrix:
    """Cotan laplacian on vertices.

    Parameters:
        mesh (SurfaceMesh): input mesh
        cotan (bool) : whether to compute real cotan values for more precise discretization or only 0/1 values as a graph laplacian. Defaults to True.
        connection (SurfaceConnectionVertices, optional): For a laplacian on 1-forms, gives the angle in local bases of all adjacent edges. Defaults to None.
        order (int, optional): Order of the parallel transport (useful when computing frame fields). Does nothing if parallel_transport is set to None. Defaults to 4.

    Returns:
        scipy.sparse.lil_matrix : the Laplacian operator as a sparse matrix
    """
    n_coeffs = 12*len(mesh.faces)
    if cotan:
        if mesh.face_corners.has_attribute("cotan"):
            cot = mesh.face_corners.get_attribute("cotan")
        else:
            cot = cotangent(mesh)
    else:
        cot = None

    rows = np.zeros(n_coeffs, dtype=np.int32)
    cols = np.zeros(n_coeffs, dtype=np.int32)
    coeffs = np.zeros(n_coeffs, dtype=(complex if connection else np.float64))
    _c = 0
    
    for iT, (p,q,r) in enumerate(mesh.faces):
        if cotan:
            a,b,c = (cot[mesh.connectivity.vertex_to_corner_in_face(_v,iT)]/2 for _v in (p,q,r))
        else:
            a,b,c = 0.5, 0.5, 0.5
        for (i, j, v) in [(p, q, c), (q, r, a), (r, p, b)]:
            rows[_c], cols[_c], coeffs[_c], _c = i, i, v, _c+1
            rows[_c], cols[_c], coeffs[_c], _c = j, j, v, _c+1
            if connection is not None:
                ai, aj = connection.transport(i,j), connection.transport(j,i)
                rows[_c], cols[_c], coeffs[_c], _c = i, j, - v * cmath.rect(1., order*(ai - aj - math.pi)), _c+1
                rows[_c], cols[_c], coeffs[_c], _c = j, i, - v * cmath.rect(1., order*(aj - ai - math.pi)), _c+1
            else:
                rows[_c], cols[_c], coeffs[_c], _c = i, j, -v, _c+1
                rows[_c], cols[_c], coeffs[_c], _c = j, i, -v, _c+1 
    
    mat = sp.csc_matrix((coeffs,(rows,cols)), dtype= (complex if connection else np.float64))
    return mat

##### For Surface, on faces #####

@allowed_mesh_types(SurfaceMesh)
def cotan_edge_diagonal(mesh : SurfaceMesh, inverse:bool=True) -> sp.csc_matrix:
    """
    Builds a diagonal matrix of size |E|x|E| where the coefficients are the cotan weights, i.e.

        M[e,e] = 1/abs( cot(a_e) + cot(b_e))

    where a_e and b_e are the opposite angles in adjacent triangles of edge e.

    Args:
        mesh (SurfaceMesh): input mesh
        inverse (bool): whether to compute M or M^-1 (all coefficients on the diagonal inverted)

    Returns:
        sp.csc_matrix
    """
    m = len(mesh.edges)
    cotan = cotangent(mesh, persistent=False)
    coeffs = np.zeros(m)
    for ie,(u,v) in enumerate(mesh.edges):
        T1,uT1,vT1 = mesh.connectivity.direct_face(u,v,True)
        T2,vT2,uT2 = mesh.connectivity.direct_face(v,u,True)
        cT1,cT2 = 0., 0.
        if T1 is not None : 
            w = mesh.faces[T1][3-uT1-vT1]
            c = mesh.connectivity.vertex_to_corner_in_face(w,T1)
            cT1 = cotan[c]
        if T2 is not None : 
            w = mesh.faces[T2][3-uT2-vT2]
            c = mesh.connectivity.vertex_to_corner_in_face(w,T2)
            cT2 = cotan[c]
        
        if inverse:
            if abs(cT1 + cT2)<1e-8:
                coeffs[ie] = 1e8
            else:
                coeffs[ie] = 1/(cT1 + cT2)
        else:
            coeffs[ie] = cT1 + cT2
    return sp.diags(coeffs, format="csc")

@allowed_mesh_types(SurfaceMesh)
def area_weight_matrix_faces(mesh : SurfaceMesh, inverse : bool=False) -> sp.csc_matrix:
    """
    Returns the diagonal matrix A of area weights on faces
    Laplace-beltrami operator for a 2D manifold is (A^-1)L where A is the area weight and L is the cotan matrix

    Args:
        mesh (SurfaceMesh): the input mesh
        inverse (bool, optional): whether to return A or A^-1. Defaults to False.

    Returns:
        sp.csc_matrix
    """
    area = face_area(mesh, persistent=False).as_array()
    if inverse:
        area = 1/area
    return sp.diags(area, format="csc")

@allowed_mesh_types(SurfaceMesh)
def laplacian_triangles(
    mesh : SurfaceMesh, 
    cotan : bool = True, 
    connection : "SurfaceConnectionFaces" = None, 
    order : int=4) -> sp.lil_matrix:
    """
    Cotan laplacian defined on face connectivity (ie on the dual mesh)

    Args:
        mesh (SurfaceMesh): the supporting mesh
        cotan (bool, optional): whether to use cotan laplacian or . Defaults to True.
        connection (SurfaceConnectionFaces, optional): _description_. Defaults to None.
        order (int, optional): _description_. Defaults to 4.

    Returns:
        scipy.sparse.lil_matrix : the Laplacian operator as a sparse matrix
    """

    n = len(mesh.faces)
    m = len(mesh.edges)

    Nabla = sp.lil_matrix((m,n), dtype=complex if connection else np.float32) # gradient matrix
    if connection is not None:
        for ie,(ei,ej) in enumerate(mesh.edges):
            T1,T2 = mesh.connectivity.edge_to_faces(ei,ej)
            if T1 is not None and T2 is not None:
                Nabla[ie,T1] = -1
                Nabla[ie,T2] = cmath.rect(1, order*connection.transport(T1,T2))
    else:
         for ie,(ei,ej) in enumerate(mesh.edges):
            T1,T2 = mesh.connectivity.edge_to_faces(ei,ej)
            if T1 is not None and T2 is not None:
                Nabla[ie,T1] = -1
                Nabla[ie,T2] = 1
    Nabla = Nabla.tocsc()
    Nabla_star = Nabla.conj().transpose()
    if cotan:
        D = cotan_edge_diagonal(mesh)
        return Nabla_star @ D @ Nabla
    else:
        return Nabla_star @ Nabla

##### For Volumes #####

@allowed_mesh_types(VolumeMesh)
def volume_weight_matrix(mesh : VolumeMesh, inverse:bool = False) -> sp.csc_matrix:
    """
    Mass diagonal matrix for volume Laplacian.
    Args:
        mesh (VolumeMesh): input mesh
        inverse (bool, optional): whether to return A or A^-1. Defaults to False.

    Returns:
        sp.csc_matrix: diagonal matrix of vertices area
    """
    A = np.zeros(len(mesh.vertices))
    volume = cell_volume(mesh, persistent=False)
    for iC,C in enumerate(mesh.cells):
        for u in C:
            A[u] += volume[iC] if not inverse else 1/volume[iC]
    return sp.diags(A, format="csc")

@allowed_mesh_types(VolumeMesh)
def volume_laplacian(mesh : VolumeMesh) -> sp.lil_matrix:
    """Volume laplacian on vertices
    
    This is the 3D extension of the cotan laplacian, ie the discretization of the Laplace-Beltrami operator on 3D manifolds.

    Parameters:
        mesh (VolumeMesh): the input mesh

    Returns:
        scipy.sparse.lil_matrix: the Laplacian operator as a sparse matrix

    References:
       [1] https://cseweb.ucsd.edu/~alchern/projects/ConformalVolume/

       [2] https://www.cs.cmu.edu/~kmcrane/Projects/Other/nDCotanFormula.pdf
    """
    if not mesh.is_tetrahedral(): 
        raise NotImplementedError
    
    n = len(mesh.vertices)
    mat = sp.lil_matrix((n,n))
    for e,(I,J) in enumerate(mesh.edges):
        omega = 0
        for ic in mesh.connectivity.edge_to_cell(e):
            K,L = (x for x in mesh.cells[ic] if x not in (I,J)) # two other vertices of the tet
            l = distance(mesh.vertices[K], mesh.vertices[L])
            _,_,Z1 = face_basis(*(mesh.vertices[_u] for _u in (I,K,L)))
            _,_,Z2 = face_basis(*(mesh.vertices[_u] for _u in (J,L,K)))
            cot = abs(dot(Z1,Z2))/norm(cross(Z1,Z2))
            omega += l * cot / 6
        mat[I,I] += omega
        mat[J,J] += omega
        mat[I,J] = -omega
        mat[J,I] = -omega
    return mat

@allowed_mesh_types(VolumeMesh)
def laplacian_tetrahedra(mesh : VolumeMesh) -> sp.csc_matrix:
    """Laplacian defined on cell connectivity (ie on the dual volume mesh)

    Parameters:
        mesh (SurfaceMesh): input mesh

    Returns:
        scipy.sparse.csc_matrix
    """
    if not mesh.is_tetrahedral():
        raise Exception("Mesh is not tetrahedral")
    n = len(mesh.cells)
    mat = sp.lil_matrix((n,n))
    for c1 in mesh.id_cells:
        mat[c1,c1] += len(mesh.connectivity.cell_to_cell(c1))
        for c2 in mesh.connectivity.cell_to_cell(c1):
            mat[c1,c2] -= 1
    return mat.tocsc()