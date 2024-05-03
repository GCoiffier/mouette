from .mesh_attributes import Attribute, ArrayAttribute
from .. import config
import warnings
from abc import ABC, abstractmethod
import numpy as np

class _BaseDataContainer(ABC):
    """
    Base class for a data container. A container encapsulate all elements of the same type in a mesh as well as attributes defined on them.
    """
    def __init__(self, attributes : dict = None, id : str = "") -> None:
        self.id = id
        if attributes is None:
            self._attr = dict()
        else:
            assert isinstance(attributes, dict)
            self._attr = attributes
    
    @abstractmethod
    def empty(self) -> bool:
        pass

    @abstractmethod
    def clear(self):
        pass

    @property
    def attributes(self):
        return self._attr.keys()

    def create_attribute(self, name, data_type, elem_size=1, dense=False, default_value=None, size=None) -> Attribute:
        # If 'name' already exists in the attribute dict, the corresponding attribute will be overridden
        if name in self._attr and not config.disable_duplicate_attribute_warning:
            warnings.warn(f"Warning ! Attribute '{name}' already exists on {self.id}")
        else:
            if dense:
                self._attr[name] = ArrayAttribute(data_type, len(self) if size is None else int(size), elem_size=elem_size, default_value=default_value)
            else:
                self._attr[name] = Attribute(data_type, elem_size=elem_size, default_value=default_value)
        return self._attr[name]

    def register_array_as_attribute(self, name, data, default_value=None):
         # If 'name' already exists in the attribute dict, the corresponding attribute will be overridden
        if name in self._attr and not config.disable_duplicate_attribute_warning:
            warnings.warn(f"Warning ! Attribute '{name}' already exists on {self.id}")
        else:
            if len(data.shape)==1: 
                data = data[:,np.newaxis] # change array of size (n,) to size (n,1)
            try:
                n_elem = data.shape[0]
                elem_size = data.shape[1]
                assert n_elem == len(self)
            except Exception as e:
                raise Exception(f"data array has invalid shape {data.shape}")
            self._attr[name] = ArrayAttribute(type(data[0,0].item()), n_elem, elem_size=elem_size, default_value=default_value)
            self._attr[name]._data = data
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

    @abstractmethod
    def append(self,val):
        pass

class DataContainer(_BaseDataContainer):
    """
    A DataContainer is a container class for all simplicial elements of the same type in an instance (for example, vertices, edges or faces).
    It stores relevant information about the combinatorics (in the _data field) as well as various attributes onto the elements (in the _attr field)
    """

    def __init__(self, data : list = None, attributes : dict = None, id : str = ""):
        super().__init__(attributes, id)
        self._data = [] if data is None else list(data)
       
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

    def __len__(self):
        return len(self._data)

    @property
    def size(self):
        return len(self._data)

    def empty(self):
        return not self._data

    def clear(self):
        self._data = []
        self._attr = dict()

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

class CornerDataContainer(_BaseDataContainer):
    """
    A CornerDataContainer is a variant of a DataContainer for corner elements, ie face corners, cell corners and cell faces. Unlike a regular data container that stores a list of simplicial elements, a corner container stores two pieces of information: the associated vertex/face of the corner and the face/cell it belongs to.
    """
    def __init__(self, elem:list = None, adj:list = None, attributes : dict = None, id : str = ""):
        super().__init__(attributes, id)
        self._elem = [] if elem is None else list(elem)
        self._adj = [] if adj is None else list(adj)

    def __getitem__(self, key:int) -> int:
        """
        Shortcut for self.elem(key)

        Args:
            key (int): corner identifier

        Returns:
            int: the vertex or the face this corner points to
        """
        return self._elem[key]

    def element(self,key:int) -> int:
        """
        Returns the mesh element (vertex or face) associated with the corner 'key'

        Args:
            key (int): identifier of a corner

        Returns:
            int: identifier of a vertex or face
        """
        return self._elem[key]

    def adj(self, key:int) -> int:
        """
        Returns the mesh element (either face or cell) from which the corner 'key' belongs to

        Args:
            key (int): identifier of a corner

        Returns:
            int: identifier of a face or a cell
        """
        return self._adj[key]

    def __iter__(self):
        return self._elem.__iter__()

    def __repr__(self):
        return list(zip(self._elem, self._adj)).__repr__()

    def __str__(self):
        return str(self._elem) + "\nAttributes: "+str(self._attr.keys())

    @property
    def size(self):
        return len(self._elem)

    def __len__(self):
        return len(self._elem)

    def empty(self):
        return not self._elem

    def clear(self):
        self._elem = []
        self._adj = []
        self._attr = dict()

    def append(self, val_elem, val_adj):
        self._elem.append(val_elem)
        self._adj.append(val_adj)
        for attr in self._attr.values():
            attr._expand(1)

    def __iadd__(self, other):
        if isinstance(other, list) or isinstance(other,tuple) or isinstance(other, set):
            for (e,a) in other:
                self._elem.append(e)
                self._adj.append(a)
            for attr in self._attr.values():
                attr._expand(len(other))
        elif isinstance(other, CornerDataContainer):
            self._elem += other._elem
            self._adj += other._adj
            for attr in self._attr.values():
                attr._expand(other.n_elem)
        else:
            raise Exception("Could not append data container of type {} onto an attribute".format(type(other)))
        return self