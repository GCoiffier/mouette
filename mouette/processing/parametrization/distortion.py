from ..worker import Worker
from ...mesh.datatypes import *
from ...mesh.mesh_attributes import Attribute, ArrayAttribute
from ...attributes.misc_faces import face_area

from ...geometry import Vec
from ... import geometry as geom
from ... import utils

import numpy as np

class ParamDistortion(Worker):
    """
    Utility class to compute various distortion metrics for surface parametrization.
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
            mesh : SurfaceMesh,
            uv_attr : str = "uv_coords",
            save_on_mesh : bool = True,
            verbose : bool = False):
        """
        Initializes the distortion utility class.

        Args:
            mesh (SurfaceMesh): the supporting mesh
            uv_attr (str, optional): the attribute name that stores the uv-coordinates on face corners. Defaults to "uv_coords".
            save_on_mesh (bool, optional): whether to store distortion values on the mesh. Defaults to True.
            verbose (bool, optional): verbose mode. Defaults to False.

        Raises:
            Exception: fails if the attribute 'uv_attr' does not exists.
        """
        super().__init__("SurfaceDistortion", verbose)
        self.mesh : SurfaceMesh = mesh
        try:
            self.UV = self.mesh.face_corners.get_attribute(uv_attr)
        except:
            self.log(f"Mesh has no attribute '{uv_attr}'. Cannot compute distortion.")
            raise Exception("Initialization failed")

        self.save_on_mesh : bool = save_on_mesh

        self._summary : dict = None

        self._conformal : ArrayAttribute = None # ||J||² / det J
        self._scale     : ArrayAttribute = None # 0.5*(det J + 1 / det J)
        self._iso       : ArrayAttribute = None # distance( (sigma_1, sigma_2), (1,1))
        self._shear     : ArrayAttribute = None # dot(c_1, c_2) of columns of jacobian 
        self._stretch   : ArrayAttribute = None # sigma_1 / sigma_2
        self._det       : ArrayAttribute = None # determinant of faces

    def run(self):
        """
        Run the distortion computation.

        Raises:
            Exception: fails if the mesh is not triangular.
            ZeroDivisionError: if degenerated elements are present in the parametrization.
        """
        if not ({len(f) for f in self.mesh.faces} == {3}) :
            # do not call mesh.is_triangular() since we would like to also be compatible with some RawMeshData
            raise Exception("Mesh is not triangular")
        self._init_containers()
        xy_area = 0.
        uv_area = 0.
        area = face_area(self.mesh, persistent=False)
        for T in self.mesh.id_faces:
            cnr = 3*T # self.mesh.connectivity.face_to_first_corner(T)
            xy_area += area[T]
            uvA,uvB,uvC = ( Vec( self.UV[cnr + i][0], self.UV[cnr + i][1], 0.) for i in range(3))
            uv_area += geom.triangle_area(uvA,uvB,uvC)

        scale_ratio = (xy_area / uv_area)

        conformalDist = 0. # ||J||² / det J
        authalicDist = 0. # det J + 1 / det J
        isoDist = 0. # distance( (sigma_1, sigma_2), (1,1))
        shearDist = 0. # dot(c_1, c_2) of columns of jacobian 
        stretchDistMean = 0 # sigma_1 / sigma_2 
        stretchDistMax = -float("inf") # sigma_1 / sigma_2

        for T in self.mesh.id_faces:
            try:
                A,B,C = self.mesh.faces[T]
                cnr = 3*T #self.mesh.connectivity.face_to_first_corner(T)
                pA,pB,pC = (self.mesh.vertices[x] for x in (A,B,C))
                X,Y,N = geom.face_basis(pA,pB,pC)

                # original coordinates of the triangle
                u0 = pB-pA
                v0 = pC-pA
                u0 = complex(X.dot(u0), Y.dot(u0))
                v0 = complex(X.dot(v0), Y.dot(v0))

                # new coordinates of the triangle
                qA,qB,qC = (self.UV[cnr + i] for i in range(3))
                u = qB - qA
                v = qC - qA
                
                # jacobian
                J0 = np.array([[u0.real, v0.real], 
                            [u0.imag, v0.imag]])
                J0 = np.linalg.inv(J0)
                J1 = np.array([[u[0], v[0]], 
                               [u[1], v[1]]])
                J = J1 @ J0

                try:
                    c1, c2 = Vec.normalized(J[:,0]), Vec.normalized(J[:,1])
                    shearDistT = abs(geom.dot(c1, c2))
                except:
                    shearDistT = 0.
                self._shear[T] = shearDistT
                shearDist += shearDistT * area[T] / xy_area

                sig = np.linalg.svd(J, compute_uv=False) # eigenvalues
                detJ = np.linalg.det(J)
                
                if abs(detJ)<1e-8:
                    raise ZeroDivisionError

                confDistT = (np.trace(np.transpose(J) * J)/detJ)/2
                self._conformal[T] = confDistT
                conformalDist += confDistT * area[T] / xy_area

                detJ *= scale_ratio
                self._det[T] = detJ
                
                authDistT = ( detJ + 1 / detJ)/2
                self._scale[T] = authDistT
                authalicDist += authDistT * area[T]/xy_area

                stretchDistT = sig[0]/sig[1]
                self._stretch[T] = stretchDistT
                stretchDistMax = max(stretchDistMax, stretchDistT)
                stretchDistMean += stretchDistT * area[T] / xy_area

                isoDistT = geom.distance( Vec(sig[0]* np.sqrt(scale_ratio), sig[1]* np.sqrt(scale_ratio)), Vec(1.,1.))
                self._iso[T] = isoDistT
                isoDist += isoDistT * area[T] / xy_area

            except ZeroDivisionError:
                continue

        self._summary = {
            "conformal" : conformalDist,
            "iso" : isoDist,
            "shear" : shearDist,
            "scale" : authalicDist,
            "stretch_mean" : stretchDistMean,
            "stretch_max" : stretchDistMax,
        }

    def _init_containers(self):
        if self.save_on_mesh:
            self._conformal = self.mesh.faces.create_attribute("conformal_dist", float, dense=True)
            self._scale = self.mesh.faces.create_attribute("scale_dist", float, dense=True)
            self._stretch = self.mesh.faces.create_attribute("stretch_dist", float, dense=True)
            self._shear = self.mesh.faces.create_attribute("shear_dist", float, dense=True)
            self._iso = self.mesh.faces.create_attribute("iso_dist", float, dense=True)
            self._det = self.mesh.faces.create_attribute("det", float, dense=True)
        else:
            N = len(self.mesh.faces)
            self._conformal = ArrayAttribute(float, N)
            self._scale = ArrayAttribute(float, N)
            self._stretch = ArrayAttribute(float, N)
            self._shear = ArrayAttribute(float, N)
            self._iso = ArrayAttribute(float, N)
            self._det = ArrayAttribute(float, N)

    @property
    def summary(self) -> dict:
        """
        Computes a summary dictionnary of all distortion values as an average over the mesh

        Returns:
            dict: a dictionnary with aggregated values over the mesh
        """
        if self._summary is None: self.run()
        return self._summary

    @property
    def conformal(self) -> ArrayAttribute:
        """
        Conformal distortion, defined as $\\frac{||J||^2}{\\det(J)}$
        """
        if self._conformal is None: self.run()
        return self._conformal
    
    @property
    def scale(self) -> ArrayAttribute:
        """
        Scale distortion, defined as $\\frac12 (\\det(J) + 1/\\det(J))$
        """
        if self._scale is None: self.run()
        return self._scale
    
    @property
    def stretch(self) -> ArrayAttribute:
        """
        Stretch distortion

        Defined as the ratio $\\frac{\\sigma_1}{\\sigma_2}$ where $\\sigma_1$ and $\\sigma_2$ are the eigenvalues of J
        """
        if self._stretch is None: self.run()
        return self._stretch

    @property
    def iso(self) -> ArrayAttribute :
        """
        Isometric distortion

        Defined as the distance from ($\\sigma_1$, $\\sigma_2$) to (1,1) where $\\sigma_1$ and $\\sigma_2$ are the eigenvalues of J
        """
        if self._iso is None: self.run()
        return self._iso

    @property
    def shear(self) -> ArrayAttribute :
        """
        Shear distortion

        Defined as $c1 \\cdot c2$ where $c1$ and $c2$ are the columns of J
        """
        if self._shear is None: self.run()
        return self._shear


class QuadQuality(Worker):
    """
    Utility class to compute various quality metrics on quad meshes
    """

    @allowed_mesh_types(SurfaceMesh)
    def __init__(self, 
            mesh : SurfaceMesh,
            save_on_mesh : bool = True,
            verbose : bool = False):
        super().__init__("SurfaceDistortion", verbose)
        self.mesh : SurfaceMesh = mesh
        self.save_on_mesh = save_on_mesh

        self._summary : dict = None
        self._conformal : ArrayAttribute = None
        self._scale : ArrayAttribute = None
        self._stretch : ArrayAttribute = None
        self._det : ArrayAttribute = None

    def run(self):
        if not ({len(f) for f in self.mesh.faces} == {4}) :
            raise Exception("Mesh is not quad")

        self._init_containers()
        ref_area = len(self.mesh.faces)
        real_area = 0.
        area = face_area(self.mesh, persistent=False)
        for T in self.mesh.id_faces:
            real_area += area[T]

        scale_ratio = (ref_area / real_area)

        conformalDist = 0. # ||J||² / det J
        scaleDist = 0. # 0.5*(det J + 1 / det J)
        detDist = 0. # det J
        stretchDistMean = 0

        for iT,T in enumerate(self.mesh.faces):
            try:
                for v1,v2,v3 in utils.cyclic_triplets(T):
                    P1,P2,P3 = (self.mesh.vertices[_v] for _v in (v1,v2,v3))
                    X,Y,_ = geom.face_basis(P1,P2,P3)
                    # new coordinates of the triangle
                    u = P1 - P2
                    v = P3 - P2
                    # jacobian
                    J = np.array([[X.dot(u), Y.dot(u)], 
                                [X.dot(v), Y.dot(v)]])

                    sig = np.linalg.svd(J, compute_uv=False)
                    detJ = np.linalg.det(J)
                    
                    if abs(detJ)<1e-8:
                        raise ZeroDivisionError

                    normJ = np.linalg.norm(J)

                    confDistT = ((normJ*normJ)/detJ)/8
                    self._conformal[iT] += confDistT
                    conformalDist += confDistT * area[iT] / real_area

                    detJ *= scale_ratio
                    detDist += detJ / (4*len(self.mesh.faces))
                    self._det[iT] += detJ/4
                    scaleDistT = ( detJ + 1 / detJ)/8

                    self._scale[iT] += scaleDistT
                    scaleDist += scaleDistT * area[iT] / real_area

                    stretchDistT = (sig[0]/sig[1])/4
                    self._stretch[iT] += stretchDistT
                    stretchDistMean += stretchDistT * area[iT] / real_area
            except ZeroDivisionError:
                continue
            except FloatingPointError:
                continue

        self._summary = {
            "conformal" : conformalDist,
            "scale" : scaleDist,
            "det" : detDist,
            "stretch" : stretchDistMean
        }
        

    def _init_containers(self):
        if self.save_on_mesh:
            self._conformal = self.mesh.faces.create_attribute("conformal_dist", float, dense=True)
            self._scale = self.mesh.faces.create_attribute("scale_dist", float, dense=True)
            self._stretch = self.mesh.faces.create_attribute("stretch_dist", float, dense=True)
            self._det = self.mesh.faces.create_attribute("det", float, dense=True)
        else:
            N = len(self.mesh.faces)
            self._conformal = ArrayAttribute(float, N)
            self._scale = ArrayAttribute(float, N)
            self._stretch = ArrayAttribute(float, N)
            self._det = ArrayAttribute(float, N)

    @property
    def summary(self):
        if self._summary is None: self.run()
        return self._summary

    @property
    def conformal(self):
        if self._conformal is None: self.run()
        return self._conformal
    
    @property
    def area(self):
        if self._scale is None: self.run()
        return self._scale
    
    @property
    def stretch(self):
        if self._stretch is None: self.run()
        return self._stretch

    @property
    def det(self):
        if self._det is None: self.run()
        return self._det

    @property
    def iso(self):
        if self._iso is None: self.run()
        return self._iso

    @property
    def shear(self):
        if self._shear is None: self.run()
        return self._shear