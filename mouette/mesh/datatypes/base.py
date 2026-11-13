from abc import ABC
from ..data_container import DataContainer, CornerDataContainer

class Mesh(ABC): 
    """The baseclass for all geometrical objects in mouette.
    The class is not supposed to be used from an outside user, but to provide a common type to PointCloud, Polyline, SurfaceMesh and VolumeMesh objects.
    The class is initialized using prebuilt datacontainers.
    """

    def __init__(self,
        vertices     : DataContainer, 
        edges        : DataContainer = None,
        faces        : DataContainer = None,
        face_corners : CornerDataContainer = None,
        cells        : DataContainer = None,
        cell_corners : CornerDataContainer = None,
        cell_faces   : CornerDataContainer = None
    ):
        # set vertices
        assert isinstance(vertices, DataContainer)
        self.vertices = vertices

        # set edges
        if edges is not None:
            assert isinstance(edges, DataContainer)
            self.edges = edges

        # set faces
        if faces is not None:
            if face_corners is None:
                raise Exception("Mesh constructor received a non empty face container without a face_corner container.")
            assert isinstance(faces, DataContainer)
            assert isinstance(face_corners, CornerDataContainer)
            self.faces = faces
            self.face_corners = face_corners

        # set cells
        if cells is not None:
            if cell_corners is None or cell_faces is None:
                raise Exception("Mesh constructor received a non empty cell container without a cell_corners or a cell_faces container")
            assert isinstance(cells, DataContainer)
            assert isinstance(cell_corners, CornerDataContainer)
            assert isinstance(cell_faces, CornerDataContainer)
            self.cells = cells
            self.cell_corners = cell_corners
            self.cell_faces = cell_faces