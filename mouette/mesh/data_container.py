from .mesh_attributes import SparseAttribute, Attribute
from .. import config
import warnings
from abc import ABC
from typing import Iterable
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

    @property
    def attributes(self):
        return self._attr.keys()

    def create_attribute(self, name: str, data_type: type, elem_size: int=1, sparse: bool=False, default_value=None) -> SparseAttribute:
        """Creates an attribute with name `name`

        Args:
            name (str): attribute's name. If 'name' already exists in the attribute dict, the corresponding attribute will be overridden. The override warning can be disabled in the config.
            data_type (type): The type of data stored in the attribute.
            elem_size (int, optional): The number of data per elements. Defaults to 1.
            sparse (bool, optional): Whether to create an `Attribute` object (numpy array storage) or a `SparseAttribute` object (dictionary storage). Defaults to False (array mode).
            default_value (Any, optional): The default value to fill the attribute with. If not provided, will be taken as the default value of the type `data_type`. Defaults to None.

        Returns:
            Attribute:
        """
        # If 'name' already exists in the attribute dict, the corresponding attribute will be overridden
        if name in self._attr and config.display_duplicate_attribute_warning:
            warnings.warn(f"Attribute '{name}' already exists on {self.id} and will be overridden")
        cls = SparseAttribute if sparse else Attribute
        self._attr[name] = cls(data_type, len(self), elem_size=elem_size, default_value=default_value)
        return self._attr[name]

    def register_array_as_attribute(self, name: str, data: np.ndarray, default_value=None):
        """Converts a numpy array as a (dense) attribute.

        Args:
            name (str): attribute's name. If 'name' already exists in the attribute dict, the corresponding attribute will be overridden. The override warning can be disabled in the config.
            data (np.ndarray): the data array. Type of the attribute will be extrapolated from `data.dtype`.
            default_value (Any, optional): The default value to fill the attribute with. If not provided, will be taken as the default value of the type `data_type`. Defaults to None.

        Raises:
            ValueError: fails if the provided array does not have a compatible (N,K) shape where N is the container's size.
        """
        # If 'name' already exists in the attribute dict, the corresponding attribute will be overridden
        if name in self._attr and config.display_duplicate_attribute_warning:
            warnings.warn(f"Attribute '{name}' already exists on {self.id} and will be overridden")
        try:
            n_elem = data.shape[0]
            elem_size = 1 if len(data.shape)==1 else data.shape[1]
            assert n_elem == len(self)
        except Exception as e:
            raise ValueError(f"data array has invalid shape {data.shape}")
        self._attr[name] = Attribute(data.dtype, n_elem, elem_size=elem_size, default_value=default_value)
        self._attr[name]._data = data
        return self._attr[name]

    def register_dict_as_sparse_attribute(self, name:str, data:dict, default_value=None):
        """Converts a dictionary as a sparse attribute. The dictionary should map integer indices to values and all values should have the same type.

        Args:
            name (str):  attribute's name. If 'name' already exists in the attribute dict, the corresponding attribute will be overridden. The override warning can be disabled in the config.
            data (dict): Type of the attribute will be extrapolated from a random element.
            default_value (Any, optional): The default value to fill the attribute with. If not provided, will be taken as the default value of the attribute's type. Defaults to None.
        """
         # If 'name' already exists in the attribute dict, the corresponding attribute will be overridden
        if name in self._attr and config.display_duplicate_attribute_warning:
            warnings.warn(f"Attribute '{name}' already exists on {self.id} and will be overridden")
        if len(data)==0:
            raise Exception("Trying to register an empty dictionary. Use `create_attribute(sparse=True)` instead.")
        dict_elem = next(iter(data.values()))
        if isinstance(dict_elem, Iterable):
            elem_size = len(dict_elem)
            dtype = type(dict_elem[0])
        else:
            elem_size = 1
            dtype = type(dict_elem)
        self._attr[name] = SparseAttribute(dtype, n_elem=len(self), elem_size=elem_size,default_value=default_value)
        self._attr[name]._data = data
        return self._attr[name]


    def delete_attribute(self, name: str):
        """Deletes the attribute associated with name `name` if it exists.

        Args:
            name (str): the attribute's name.
        """
        if name in self._attr:
            del self._attr[name]

    def has_attribute(self, name: str) -> bool:
        """Returns True is an attribute with name `name` is defined for this container.

        Args:
            name (str): the attribute's name

        Returns:
            bool:
        """
        return name in self._attr

    def get_attribute(self, name: str) -> SparseAttribute :
        """Returns the attribute with name `name`. Fails if the attribute does not exist

        Args:
            name (str): the attribute's name

        Raises:
            KeyError: no attribute has name `name` inside this data container.

        Returns:
            Attribute: the attribute
        """
        if name not in self._attr:
            raise KeyError("Attribute does not exist")
        return self._attr[name]



class DataContainer(_BaseDataContainer):
    """
    A `DataContainer` is a container class for all elements of the same type in an instance (for example, vertices, edges or faces).
    It stores relevant information about the combinatorics (in the `_data` field) as well as various attributes onto the elements (in the `_attr` field)
    """

    def __init__(self, data: Iterable = None, attributes : dict = None, id : str = ""):
        super().__init__(attributes, id)
        self.set_data(data)
       
    def set_data(self, data: Iterable):
        if (data is None) or \
        (isinstance(data, np.ndarray) and data.size==0) or \
        (isinstance(data, Iterable) and len(data)==0): 
            self._data = np.empty(0)
            return
        
        if isinstance(data, np.ndarray):
            dtype = data.dtype
        elif data is not None:
            try:
                dtype = type(data[0][0])
            except AttributeError:
                dtype = type(data[0])
        else:
            dtype=None
        self._data = np.ascontiguousarray(data, dtype=dtype)

    def clear(self):
        self._data = np.empty(0)
        self._attr = dict()

    def empty(self):
        return self._data.size==0

    def __getitem__(self, key):
        if isinstance(key, DataContainer):
            # so we can write `mesh.vertices[mesh.faces]` for instance
            return self._data[key._data]
        return self._data[key]

    def __setitem__(self, key, value):
        try:
            self._data[key] = value
        except Exception as e:
            warnings.warn(f"Error in attribute '{key}' : {e}")
            raise Exception("Aborting")

    def __iter__(self):
        return self._data.__iter__()
        # return self._data.tolist().__iter__()

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return str(self._data) + "\nAttributes: "+str(self._attr.keys())

    def __len__(self):
        return self._data.shape[0]

    def __array__(self, dtype=None, copy=None):
        if dtype is None or np.dtype(dtype) == self._data.dtype:
            return self._data.copy() if copy else self._data
        if copy is False:
            raise ValueError(f"Unable to avoid copy while casting from {self._data.dtype} to {np.dtype(dtype)}")
        return self._data.astype(dtype) # astype already returns a new array

    @property
    def size(self) -> int:
        """The container size. Alias for `len(container)`

        Returns:
            int: number of elements in the container
        """
        return self.__len__()



class CornerDataContainer(_BaseDataContainer):
    """
    A `CornerDataContainer` is a variant of a DataContainer for corner elements. It is used for face corners, cell corners and cell faces. 
    
    Unlike a regular data container that stores a list of simplicial elements, a corner container stores two pieces of information: the associated vertex/face of the corner and the face/cell it belongs to.
    """
    def __init__(self, elem:Iterable = None, adj:Iterable = None, attributes : dict = None, id : str = ""):
        super().__init__(attributes, id)
        self.set_data(elem, adj)

    def set_data(self, elem, adj):
        self._elem = np.empty(0) if elem is None else np.ascontiguousarray(elem)
        self._adj = np.empty(0) if elem is None else np.ascontiguousarray(adj)

    def clear(self):
        self._elem = np.empty(0)
        self._adj = np.empty(0)
        self._attr = dict()

    def empty(self):
        return self._elem.size==0

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
        # return self._elem.tolist().__iter__()
        return self._elem.__iter__()

    def __repr__(self):
        return list(zip(self._elem, self._adj)).__repr__()

    def __str__(self):
        return str([x for x in zip(self._elem, self._adj)]) + "\nAttributes: "+str(self._attr.keys())

    def __array__(self, dtype=None, copy=None):
        if dtype is None or np.dtype(dtype) == self._elem.dtype:
            return self._elem.copy() if copy else self._elem
        if copy is False:
            raise ValueError(f"Unable to avoid copy while casting from {self._elem.dtype} to {np.dtype(dtype)}")
        return self._elem.astype(dtype) # astype already returns a new array

    @property
    def size(self):
        """The container size. Alias for `len(container)`

        Returns:
            int: number of elements in the container
        """
        return self.__len__()

    def __len__(self):
        return self._elem.shape[0]