from .mesh_attributes import Attribute, ArrayAttribute
from ..geometry import Vec
from .. import utils
from .. import config
import itertools
import warnings
from .. import config

class DataContainer:
    """
    A DataContainer is a container class for all simplicial elements of the same type in an instance (for example, vertices, edges or faces).
    It stores relevant information about the combinatorics (in the _data field) as well as various attributes onto the elements (in the _attr field)
    """

    def __init__(self, data : list = None, attributes : dict = None, id : str = ""):
        self.id = id
        if data is None:
            self._data = []
        else:
            self._data = list(data)
        
        # self._attr : dict of str->Attribute
        if attributes is None:
            self._attr = dict()
        else:
            assert isinstance(attributes, dict)
            self._attr = attributes 
    
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        try:
            self._data[key] = value
        except Exception as e:
            warnings.warn(f"Error in attribute '{key}' : {e}")
            raise Exception("Aborting")

    def __iter__(self):
        return self._data.__iter__()

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return str(self._data) + "\nAttributes: "+str(self._attr.keys())

    @property
    def size(self):
        return len(self._data)

    def __len__(self):
        return len(self._data)

    def empty(self):
        return not self._data

    def clear(self):
        self._data = []
        self._attr = dict()

    @property
    def attributes(self):
        return self._attr.keys()

    def create_attribute(self, name, data_type, elem_size=1, dense=False, default_value=None) -> Attribute:
        # If 'name' already exists in the attribute dict, the corresponding attribute will be overridden
        if name in self._attr and not config.disable_duplicate_attribute_warning:
            warnings.warn(f"Warning ! Attribute '{name}' already exists on {self.id}")
        else:
            if dense:
                self._attr[name] = ArrayAttribute(data_type, len(self), elem_size=elem_size, default_value=default_value)
            else:
                self._attr[name] = Attribute(data_type, elem_size=elem_size, default_value=default_value)
        return self._attr[name]

    def delete_attribute(self, name):
        if name in self._attr:
            del self._attr[name]

    def has_attribute(self, name) -> bool:
        return name in self._attr

    def get_attribute(self, name) -> Attribute :
        if name not in self._attr:
            raise Exception("Attribute does not exist")
        return self._attr[name]

    def append(self,val):
        self._data.append(val)
        for attr in self._attr.values():
            attr._expand(1)

    def __iadd__(self,other):
        if isinstance(other, list) or isinstance(other,tuple) or isinstance(other, set):
            self._data += list(other)
            for attr in self._attr.values():
                attr._expand(len(other))
        elif isinstance(other,DataContainer):
            self._data += other._data
            for attr in self._attr.values():
                attr._expand(other.n_elem)
        else:
            raise Exception("Could not append data container of type {} onto an attribute".format(type(other)))
        return self


class RawMeshData:
    """
    A base container class to store all the data relative to a mesh.
    It is not made to be used outside of the library, and serves an intermediate between io parsers and instantiated (and typed) mesh classes.
    """

    def __init__(self, mesh=None):
        self.vertices = DataContainer(id="vertices") if mesh is None else mesh.vertices
        self.edges = DataContainer(id="edges") if (mesh is None or not hasattr(mesh,"edges")) else mesh.edges

        self.faces = DataContainer(id="faces") if (mesh is None or not hasattr(mesh,"faces")) else mesh.faces
        self.face_corners = DataContainer(id="face_corners") if (mesh is None or not hasattr(mesh,"face_corners")) else mesh.face_corners
       
        self.cells = DataContainer(id="cells") if (mesh is None or not hasattr(mesh,"cells")) else mesh.cells
        self.cell_corners = DataContainer(id="cell_corners") if (mesh is None or not hasattr(mesh,"cell_corners")) else mesh.cell_corners
        self.cell_faces = DataContainer(id="cell_faces") if (mesh is None or not hasattr(mesh,"cell_faces")) else mesh.cell_faces

        self._dimensionality : int = None
        self._prepared : bool = False
    
    @property
    def id_vertices(self):
        """
        Shortcut for range(len(self.vertices))
        """
        return range(len(self.vertices))

    @property
    def id_edges(self):
        """
        Shortcut for range(len(self.edges))
        """
        return range(len(self.edges))

    @property
    def id_faces(self):
        """
        Shortcut for range(len(self.faces))
        """
        return range(len(self.faces))

    @property
    def id_cells(self):
        """
        Shortcut for range(len(self.cells))
        """
        return range(len(self.cells))

    @property
    def dimensionality(self) -> int:
        """
        Dimensionality of the manifold represented by the data.
        If only vertices -> 0
        If vertices and edges (embedded graph) -> 1
        If faces (surface manifold) -> 2
        If cells (volume manifold) -> 3

        Returns:
            int
        """
        if self._dimensionality is None:
            self._compute_dimensionality()
        return self._dimensionality

    def _compute_dimensionality(self):
        if not self.cells.empty():
            self._dimensionality = 3
        elif not self.faces.empty():
            self._dimensionality = 2
        elif not self.edges.empty():
            self._dimensionality = 1
        else:
            self._dimensionality = 0

    def prepare(self) -> None:
        """
        Prepares the data to have the correct format.
        On vertices : casts to M.Vec
        On edges : sorts edge tuples to satisfy edge convention (smallest index first)
        On faces : nothing
        On cells : nothing

        if option is set in config, adds to the edge container all the edges implicitly defined by the faces
        """
        if self._prepared : return

        # call completion before preparation
        if config.complete_faces_from_cells:
            self._complete_faces_from_cells()

        # call edge completion ***after*** face completion
        if config.complete_edges_from_faces : 
            self._complete_edges_from_faces() 

        self._prepare_vertices()
        self._prepare_edges()
        self._prepare_faces()
        self._prepare_face_corners()
        self._prepare_cells()
        self._compute_dimensionality()
        self._prepared = True

    def _prepare_vertices(self):
        for iv in self.id_vertices:
            self.vertices[iv] = Vec(self.vertices[iv])

    def _prepare_edges(self):
        N = len(self.vertices)

        def is_valid(a,b):
            return a!=b and 0<=a<N and 0<=b<N

        # Filter invalid edges (like (x,x)) and make the other immutable
        edges_invalid = any((not is_valid(a,b) for a,b in self.edges))
        if edges_invalid:
            # Rebuild the edge container
            new_edges = DataContainer(id="edges")
            new_attrs = dict()
            old_attrs = dict()
            for attr_name in self.edges.attributes:
                old_attrs[attr_name] = self.edges.get_attribute(attr_name)
                new_attrs[attr_name] = new_edges.create_attribute(attr_name, old_attrs[attr_name].type, old_attrs[attr_name].elemsize)
            n = 0
            for ie in self.id_edges:
                a,b = self.edges[ie]
                if is_valid(a,b):
                    new_edges.append(utils.keyify(a,b))
                    for name in new_attrs:
                        if ie in old_attrs[name]: # keep sparsity of the attribute
                            new_attrs[name][n] = old_attrs[name][ie]
                    n+=1
            self.edges = new_edges
        else:
            for ie in self.id_edges:
                self.edges[ie] = utils.keyify(self.edges[ie])

    def _prepare_faces(self):
        pass
        # for iF in self.id_faces:
        #     self.faces[iF] = tuple(self.faces[iF])
        

    def _prepare_face_corners(self):
        nc = len(self.face_corners)
        nf = sum([len(f) for f in self.faces])
        if nc == 0 or nc!= nf : 
            # corners were not generated
            self.face_corners._data = []
            for f in self.faces:
                self.face_corners._data += list(f)

    def _prepare_cells(self):
        pass

    def _complete_edges_from_faces(self):
        if self.faces.empty() : return # nothing to do
        
        # Create hard edges attribute for already existing edges
        hard_edges = self.edges.create_attribute("hard_edges", bool)
        for e in self.id_edges: hard_edges[e] = True
        
        edge_set = set([utils.keyify(e) for e in self.edges])
        for f in self.faces:
            nf = len(f)
            for i in range(nf):
                edge = utils.keyify(f[i], f[(i+1)%nf])
                if edge not in edge_set:
                    edge_set.add(edge)
                    self.edges.append(edge)

    def _complete_faces_from_cells(self):
        if self.cells.empty() : return # nothing to do
        face_set = set([utils.keyify(f) for f in self.faces])
        for C in self.cells:
            if len(C)>4:
                continue
            for F in itertools.combinations(C,3): # all possibilities of 3-uples among the tetrahedron
                face = utils.keyify(F)
                if face not in face_set:
                    face_set.add(face)
                    self.faces.append(F)
