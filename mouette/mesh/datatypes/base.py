from ..mesh_data import RawMeshData

class Mesh: 
    
    def __init__(self, dim : int, data : RawMeshData = None):
        if data is None:
            data = RawMeshData()
        else:
            data.prepare()

        self.vertices = data.vertices

        if dim>0:
            self.edges = data.edges
        
        if dim>1:
            self.faces = data.faces
            self.face_corners = data.face_corners

        if dim>2:
            self.cells = data.cells
            self.cell_corners = data.cell_corners
            self.cell_faces = data.cell_faces
