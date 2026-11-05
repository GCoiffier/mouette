import numpy as np
from abc import abstractmethod, ABC
from aenum import MultiValueEnum


################ Typing ################
class AttributeType(MultiValueEnum):
    """
    Typing module for attributes. The main goal of this class is to prevent automatic casting that usually happens in python.
    """
    Bool = bool, np.bool_
    Int = int, np.int32, np.uint8, np.int64
    Float = float, np.float32, np.float64, np.float128
    Complex = complex, np.complex64, np.complex128

    @classmethod
    def from_string(cls, txt : str):
        if txt in {"complex", "\"complex\""}:
            return cls.Complex
        if txt in {"double", "\"double\"", "float", "\"float\""}:
            return cls.Float
        if txt in {"index_t", "\"index_t\"", "int", "\"int\"", "\"signed_index_t\""}:
            return cls.Int
        if txt in {"bool", "\"bool\""}:
            return cls.Bool
        raise Exception(f"String '{txt}' corresponds to no attribute type")

    def to_string(self):
        if self.name.lower()=="float":
            return "double"
        return self.name.lower()

    def byte_size(self):
        return {
            "Bool" : 1,
            "Int" : 4,
            "Float" : 8
        }.get(self.name, None)

    @property
    def dtype(self):
        return self.value

    def default_value(self,size=1):
        if size==1:
            if self == AttributeType.Bool: return False
            if self == AttributeType.Int: return int(0)
            if self == AttributeType.Float: return float(0.)
            if self == AttributeType.Complex : return complex(0.,0.)
            raise _BaseAttribute.InvalidTypeError(self)
        return np.full(size, fill_value=self.default_value(1))

    @staticmethod
    def can_be_casted(ta : "AttributeType", tb : "AttributeType"):
        """ Check for type compatibility and casting """
        if ta==tb : return True
        casts = {(AttributeType.Bool, AttributeType.Int),
                 (AttributeType.Bool, AttributeType.Float),
                 (AttributeType.Int,AttributeType.Float)}
        return (ta,tb) in casts



class _BaseAttribute(ABC):

    ########################################

    class InvalidTypeError(Exception):
        def __init__(self, typ):
            message = f"Type '{typ}' is not a valid type for a mesh attribute"
            super().__init__(message)

    class InvalidSizeError(Exception):
        def __init__(self, n,m):
            message = f"Attempted to set an attribute value of {n} elements on an attribute of element size {m}"
            super().__init__(message)

    class TypeNotMatchingError(Exception):
        def __init__(self, val, t1, t2):
            message = f"Tried to set value {val} of type '{t1}' inside an attribute of type '{t2}'"
            super().__init__(message)

    class DefaultValueTypeDoesNotMatchError(Exception):
        def __init__(self, dflt_val, typ):
            message = f"Default value '{dflt_val}' has type '{type(dflt_val)}'' while attribute has type '{typ}'"
            super().__init__(message)

    
    ########################################

    def __init__(self, elem_type: type, n_elem:int, elem_size:int, default_value, is_sparse:bool):
        """
        the _BaseAttribute class is not supposed to be manipulated outside of the DataContainer class

        Parameters:
            elem_type (type): the type of the attribute (bool, int, float, complex)
            
            n_elem (int): Total number of elements in the container. Should match the size of the DataContainer the attribute is stored in.

            elem_size (int): Number of elem_type objects to be stored per element.
            
            default_value : the default value of the attribute is n_elem is not specified. 
                If it is not specified either, it will correspond to the default value of the type provided in elem_type.

            is_sparse (bool): whether the attribute is sparse (based on python dict) or dense (based on numpy array).
        """
        self.type: AttributeType = AttributeType(elem_type)
        self.n_elem: int = n_elem
        self.elem_size : int = elem_size
        self._default_value = default_value
        self._check_default_value_type()
        self._data = None
        self._is_sparse: bool = is_sparse

    @property
    def is_sparse(self) -> bool:
        """
        Returns:
            bool: whether the attribute is sparse (based on python dict) or dense (based on numpy array).
        """
        return self._is_sparse
    
    @property
    def default_value(self):
        if self._default_value is None:
            self._default_value = self.type.default_value(self.elem_size)
        return self._default_value

    def _check_default_value_type(self):
        if self._default_value is None : return
        try:
            default_value_type = AttributeType(type(self._default_value))
        except ValueError:
            raise _BaseAttribute.DefaultValueTypeDoesNotMatchError(self._default_value, self.type)
        if default_value_type != self.type:
            raise _BaseAttribute.DefaultValueTypeDoesNotMatchError(self._default_value, self.type)

    def _check_out_of_bounds(self,key):
        if key<0 or key>=self.n_elem:
            raise IndexError(f"Index {key} is out of bounds for attribute with size {self.n_elem}")

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return str(self._data)

    @abstractmethod
    def __array__(self, dtype=None, copy=None):
        return

    @abstractmethod
    def __iter__(self):
        pass

    def as_array(self, dtype=None, copy=None):
        """Returns a numpy array containing the data of the attribute. Similar to `np.asarray(attr)`
        """
        return self.__array__(dtype, copy)

    @abstractmethod
    def clear(self):
        """
        Empties the attribute. Frees the memory and ensures that all access return default value
        """
        pass



class Attribute(_BaseAttribute):
    def __init__(self, elem_type:type, n_elem:int, elem_size:int=1, default_value=None):
        """
        __init__ method and the whole Attribute class are not supposed to be manipulated outside of the DataContainer class.
        An ArrayAttribute stores its values in a numpy array. This is a less flexible but safer approach than Attribute.

        Parameters:
            elem_type : the type of the attribute (bool, int, float, complex)

            n_elem (int): Total number of elements in the container. Should match the size of the DataContainer the attribute is stored in.

            elem_size (int, optional): Number of elem_type objects to be stored per element. Defaults to 1.
            
            default_value (optional) : the default value of the attribute is n_elem is not specified. 
                If it is not specified either, it will correspond to the default value of the type provided in elem_type.
        """
        super().__init__(elem_type, n_elem, elem_size, default_value, False)
        if self.elem_size==1: 
            self._data = np.full(n_elem, self.default_value, dtype=self.type.dtype)
        else:
            self._data = np.full((n_elem, elem_size), self.default_value, dtype=self.type.dtype)

    def __array__(self, dtype=None, copy=None):
        if copy:
            return self._data.copy().astype(dtype)
        return self._data.astype(dtype)

    def __getitem__(self, key):
        self._check_out_of_bounds(key) #check because key=-1 returns a correct result with numpy while we want to avoid this.
        return self._data[key]
                    
    def __setitem__(self, key, value):
        self._check_out_of_bounds(key)
        if self.elem_size==1:
            datatype = type(value)
            data_attr_type = AttributeType(datatype)
            if not AttributeType.can_be_casted(data_attr_type, self.type):
                raise self.TypeNotMatchingError(value, datatype, self.type)
            self._data[key] = value
        else:
            data = np.ascontiguousarray(value)
            datatype = AttributeType(type(data[0]))
            if not AttributeType.can_be_casted(datatype, self.type):
                raise self.TypeNotMatchingError(data, data.dtype, self.type)
            if data.shape[0]!= self.elem_size:
                # provided number of element do not match elemsize
                raise self.InvalidSizeError(data.shape[0], self.elem_size)
            self._data[key] = data

    def __delitem__(self, key):
        if self.elem_size==1:
            self._data[key] = self.default_value
        else:
            self._data[key] = np.full(self.elem_size, self.default_value, dtype=self.type.dtype)

    def __len__(self):
        return self.n_elem

    def __iter__(self):
        for i in range(self.n_elem):
            yield self._data[i]

    def _expand(self, n:int):
        self.n_elem += n
        if self.elem_size==1:
            new_data = np.full(n, self.default_value, dtype=self.type.dtype)
        else:
            new_data = np.full((n, self.elem_size), self.default_value, dtype=self.type.dtype)
        self._data = np.concatenate((self._data, new_data))

    def clear(self):
        """
        Empties the attribute. Frees the memory and ensures that all access return default value
        """
        if self.elem_size==1: 
            self._data = np.full(self.n_elem, self.default_value, dtype=self.type.dtype)
        else:
            self._data = np.full((self.n_elem, self.elem_size), self.default_value, dtype=self.type.dtype)



class SparseAttribute(_BaseAttribute):
    """
    A SparseAttribute is a container based on a python dict. It allows to store data for each element of a mesh (vertices, edges, faces, cells) in a _sparse_ way, meaning that only the assigned elements get a value stored in the dictionary, and every other element falls back to a default specified value.
    """
    def __init__(self, elem_type:type, n_elem: int, elem_size:int=1, default_value=None):
        """
        __init__ method is not supposed to be manipulated outside of the DataContainer class

        Parameters:
            elem_type : the type of the attribute (bool, int, float, complex)
            
            n_elem (int): Total number of elements in the container. Should match the size of the DataContainer the attribute is stored in.

            elem_size (int, optional): Number of elem_type objects to be stored per element. Defaults to 1.
            
            default_value (optional) : the default value of the attribute is n_elem is not specified. 
                If it is not specified either, it will correspond to the default value of the type provided in elem_type.
        """
        super().__init__(elem_type, n_elem, elem_size, default_value, True)
        self._data = dict()
    
    def __getitem__(self, key):
        self._check_out_of_bounds(key)
        if key in self._data:
            return self._data[key]
        dflt = self.default_value
        # return a copy for vector defaults so that mutating the result does not alter the shared default
        return dflt.copy() if self.elem_size > 1 else dflt

    def __setitem__(self, key, value):
        self._check_out_of_bounds(key)
        if self.elem_size>1:
            data = list(value)
            n = len(data)
            if not n==self.elem_size: 
                # provided number of element do not match elemsize
                raise self.InvalidSizeError(n,self.elem_size)
        
            datatype = type(data[0])
            data_attr_type = AttributeType(datatype)  
            if not AttributeType.can_be_casted(data_attr_type, self.type):
                raise self.TypeNotMatchingError(data, datatype, self.type)
            self._data[key] = np.asarray(data)
        else:
            datatype = type(value)
            data_attr_type = AttributeType(datatype)
            if not AttributeType.can_be_casted(data_attr_type, self.type):
                raise self.TypeNotMatchingError(value, datatype, self.type)
            self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return self.n_elem

    def __iter__(self):
        """
        Sparse attributes allow to iterate only over non-default elements
        """
        return self._data.keys().__iter__()

    def __array__(self, dtype=None, copy=None):
        if self.elem_size == 1:
            out = np.full(self.n_elem, self.default_value, dtype= self.type.dtype)
        else:
            out = np.full((self.n_elem, self.elem_size), self.default_value, dtype= self.type.dtype)
        for i,x in self._data.items():
            out[i] = x
        return out.astype(dtype)

    def _expand(self, n:int):
        self.n_elem += n
    
    def clear(self):
        """
        Empties the attribute. Frees the memory and ensures that all access return default value
        """
        self._data = dict()
