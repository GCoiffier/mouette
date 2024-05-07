from ...mesh.datatypes import *
from ...mesh.mesh_attributes import Attribute
from ..features import FeatureEdgeDetector
from ..connection import SurfaceConnection
from ...utils.argument_check import *

# 2D implementations
from .faces2d import *
from .vertex2d import *
from .curvature_vertex import PrincipalDirectionsVertices
from .curvature_faces import PrincipalDirectionsFaces

# 3D implementations
from .vertex3d import *
from .cells import *

@allowed_mesh_types(SurfaceMesh)
def SurfaceFrameField(
    mesh : SurfaceMesh,
    elements : str,
    order : int = 4,
    features : bool = True,
    verbose : bool = False,
    n_smooth : int = 3,
    smooth_attach_weight : float = None,
    use_cotan : bool = True,
    cad_correction : bool = True,
    smooth_normals : bool = True,
    singularity_indices : Attribute = None,
    custom_connection : SurfaceConnection = None,
    custom_features : FeatureEdgeDetector = None
) -> FrameField :
    """
    Framefield implementation selector.

    Args:
        mesh (SurfaceMesh): the supporting mesh onto which the framefield is based

        elements (str): "vertices" or "faces", the mesh elements onto which the frames live.
    
    Keyword Args:
        order (int, optional): Order of the frame field (number of branches). Defaults to 4.
        
        features (bool, optional): Whether to consider feature edges or not. 
            If no 'custom_features' argument is provided, features will be automatically detected (see the FeatureEdgeDetector class). Defaults to True.
        
        n_smooth (int, optional): Number of smoothing steps to perform. Defaults to 3.
        
        smooth_attach_weight (float, optional): Custom attach weight to previous solution during smoothing steps. 
            If not provided, will be estimated automatically during optimization. Defaults to None.

        use_cotan (bool, optional): whether to use cotan for a better approximation of the Laplace-Beltrami operator. 
            If False, will use a simple adjacency laplacian operator (See the _operators_ module). Defaults to True.

        cad_correction (bool, optional): Whether to modify the parallel transport as in [2] to prevent singularities to appear close to pointy areas. 
            Will overwrite any connection provided with the 'custom_connection' argument. Defaults to True.

        smooth_normals : Whether to initialize the frame field as a mean of adjacent feature edges (True), or following one of the edges (False). has no effect for frame field on faces. Defaults to True.

        verbose (bool, optional): verbose mode. Defaults to False.
        
        singularity_indices (Attribute, optional): custom singularity indices for the frame field. If provided, will use the algorithm described in [3] to get the smoothest frame field with these singularities.
            If elements is "vertices", the attribute should be indexed by the faces (where singularities appear)
            If elements is "faces", the attribute should be indexed by the vertices
            /!\\ Indices should respect the Poincarré-Hopf theorem. Defaults to None.
        
        custom_connection (SurfaceConnection, optional): custom connection object to be used for parallel transport. If not provided, a connection will be automatically computed (see SurfaceConnection class). Defaults to None.
        
        custom_feature (FeatureEdgeDetector, optional): custom feature edges to be used in frame field optimization. If not provided, feature edges will be automatically detected. If the 'features' flag is set to False, features of this object are ignored. Defaults to None.

    Raises:
        InvalidRangeArgumentError: 'order' should be >= 1
        InvalidRangeArgumentError: 'n_smooth' should be >= 0
        InvalidRangeArgumentError: 'smooth_attach_weight' should be >= 0

    Returns:
        FrameField: A framefield object with the correct specifications

    References: 
        - [1] _An Approach to Quad Meshing Based on Harmonic Cross-Valued Maps and the Ginzburg-Landau Theory_, Viertel and Osting (2018)
        
        - [2] _Frame Fields for CAD models_, Desobry et al. (2022)
        
        - [3] _Trivial Connections on Discrete Surfaces_, Crane et al. (2010)
    """

    ### Assert sanity of arguments
    check_argument("elements", elements, str, ["vertices", "faces"])
    check_argument("order", order, int)
    if order<1: raise InvalidRangeArgumentError("order",order, ">=1")  
    check_argument("n_smooth", n_smooth, int)
    if n_smooth<0: raise InvalidRangeArgumentError("n_smooth", n_smooth, ">=0")
    if smooth_attach_weight is not None and smooth_attach_weight<=0 :
        raise InvalidRangeArgumentError("smooth_attach_weight", smooth_attach_weight, ">=0")

    ### Build the correct FF class
    if elements=="vertices":
        if singularity_indices is not None:
            return TrivialConnectionVertices(mesh, singularity_indices, order, verbose, use_cotan=use_cotan, cad_correction=cad_correction, custom_connection=custom_connection, custom_features=custom_features)
        else:
            return FrameField2DVertices(mesh, order, features, verbose, n_smooth=n_smooth, smooth_attach_weight=smooth_attach_weight, 
            use_cotan=use_cotan, cad_correction=cad_correction, smooth_normals=smooth_normals, custom_connection=custom_connection, custom_features=custom_features)

    elif elements=="faces":
        if singularity_indices is not None:
            return TrivialConnectionFaces(mesh, singularity_indices, order=order, verbose=verbose, custom_connection=custom_connection, custom_features=custom_features)
        return FrameField2DFaces(mesh, order, features, verbose, n_smooth=n_smooth, smooth_attach_weight=smooth_attach_weight,use_cotan=use_cotan,custom_connection=custom_connection,custom_features=custom_features)

@allowed_mesh_types(SurfaceMesh)
def PrincipalDirections(
    mesh : SurfaceMesh, 
    elements : str,
    features : bool = True,
    verbose : bool = False,
    n_smooth : int = 1,
    smooth_attach_weight : float = None,
    patch_size : int = 3,
    curv_threshold : float = 0.01,
    complete_ff : bool = True,
    custom_features : FeatureEdgeDetector = None) -> FrameField:
    """
    Principal curvature directions as a frame field.

    Args:
        mesh (SurfaceMesh): the supporting mesh onto which the framefield is based

        elements (str): "vertices" or "faces", the mesh elements onto which the frames live.
    
    Keyword Args:
        features (bool, optional): Whether to consider feature edges or not. 
            If no 'custom_features' argument is provided, features will be automatically detected (see the FeatureEdgeDetector class). Defaults to True.

        verbose (bool, optional): verbose mode. Defaults to False.

        n_smooth (int, optional): Number of smoothing steps to perform. Defaults to 1.
        
        smooth_attach_weight (float, optional): Custom attach weight to previous solution during smoothing steps. 
            If not provided, will be estimated at 1 for vertex version and 1e-3 for faces version. Defaults to None.
        
        patch_size (int, optional): On vertices only. Radius (in nubmer of edges) of the neighboring patch to be considered to approximate the shape operator. Defaults to 3.
        
        curv_threshold (float, optional): On vertices only. Threshold on the eigenvalues of the shape operator. If smaller, eigenvectors will not be extracted (curvature is too small). Defaults to 0.01.
        
        complete_ff (bool, optional): Whether to smoothly interpolate directions on low curvature areas (< curv_threshold). Defaults to True.

        custom_features (FeatureEdgeDetector, optional): custom feature edges to be used in frame field optimization. If not provided, feature edges will be automatically detected. If the 'features' flag is set to False, features of this object are ignored. Defaults to None.

    Returns:
        Framefield : a frame field object representing the curvature directions

    References:
        - [1] [https://en.wikipedia.org/wiki/Principal_curvature](https://en.wikipedia.org/wiki/Principal_curvature)
        
        - [2] _Restricted Delaunay Triangulations and Normal Cycle_, Cohen-Steiner and Morvan (2003)
    """
    ### Assert sanity of arguments
    check_argument("elements", elements, str, ["vertices", "faces"])

    ### Build the correct FF class
    if elements=="vertices":
        return PrincipalDirectionsVertices(mesh, features, verbose,
        patch_size=patch_size,n_smooth=n_smooth, smooth_attach_weight=smooth_attach_weight, 
        curv_threshold=curv_threshold,complete_ff=complete_ff, custom_features=custom_features)
    elif elements=="faces":
        return PrincipalDirectionsFaces(mesh, features, verbose, 
        n_smooth=n_smooth, smooth_attach_weight=smooth_attach_weight, custom_features=custom_features)

@allowed_mesh_types(VolumeMesh)
def VolumeFrameField(
    mesh : VolumeMesh,
    elements : str,
    features : bool = True,
    n_smooth : int = 3,
    smooth_attach_weight : float = None, 
    verbose : bool = False,
    custom_boundary_features : FeatureEdgeDetector = None) -> FrameField:
    """
    3D version of frame fields, implemented as L4 Spherical Harmonics.

    Args:
        mesh (VolumeMesh): the supporting mesh

        elements (str): "vertices" or "cells", the mesh elements onto which the frames live.

    Keyword Args:
        features (bool, optional): Whether to consider feature edges or not. Has no effect on the cell implementation. Defaults to True.

        n_smooth (int, optional): Number of smoothing steps to perform. Defaults to 1.
        
        smooth_attach_weight (float, optional): Custom attach weight to previous solution during smoothing steps. 
            If not provided, will be estimated automatically before optimization. Defaults to None.

        verbose (bool, optional): verbose mode. Defaults to False.

        custom_boundary_features (FeatureEdgeDetector, optional): custom feature edges to be used in frame field optimization. If not provided, feature edges will be automatically detected. If the 'features' flag is set to False, features of this object are ignored. Has no effect on the cell implementation. Defaults to None.

    Returns:
        Framefield: a frame field object with the correct settings
    
    References:
        - [1] _Practical 3D frame field generation_, Ray et al. (2016)
    """

    check_argument("elements", elements, str, ["vertices", "cells"])
    
    if elements=="vertices":
        return FrameField3DVertices(mesh, features, verbose, 
        n_smooth=n_smooth, smooth_attach_weight=smooth_attach_weight, 
        custom_boundary_features=custom_boundary_features)
    elif elements=="cells":
        return FrameField3DCells(mesh, verbose,
        n_smooth=n_smooth, smooth_attach_weight=smooth_attach_weight)