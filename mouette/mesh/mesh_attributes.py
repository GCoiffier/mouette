import numpy as np
from abc import abstractmethod, ABC

from ..geometry import Vec
from aenum import MultiValueEnum

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

    class OutOfBoundsError(Exception):
        def __init__(self,n, attr_size):
            message = f"Index {n} is out of bounds for attribute with size {attr_size}"
            super().__init__(message)

    ################ Typing ################

    SUPPORTED_TYPES = {
        bool, np.bool_,
        int, np.uint8, np.int32, np.int64, 
        float, np.float32, np.float64, 
        complex, 
        str
    }

    class Type(MultiValueEnum):
        Bool = bool, np.bool_
        Int = int, np.int32, np.uint8, np.int64
        Float = float, np.float32, np.float64
        Complex = complex
        String = str

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
            if txt in {"str", "string"}:
                return cls.String
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
            if self == Attribute.Type.String : 
                return "<U32" # String are limited to 32 characters max
            return self.value

        def default_value(self,n=1):
            if n==1:
                if self == Attribute.Type.Bool: return False
                if self == Attribute.Type.Int: return int(0)
                if self == Attribute.Type.Float: return float(0.)
                if self == Attribute.Type.Complex : return complex(0.,0.)
                if self == Attribute.Type.String : return ""
                raise Attribute.InvalidTypeError(self)
            return Vec([self.default_value(1)]*n)

    @staticmethod
    def _can_be_casted(ta : Type, tb : Type):
        """ Check for type compatibility and casting """
        if ta==tb : return True
        casts = {(Attribute.Type.Bool, Attribute.Type.Int),
                 (Attribute.Type.Bool, Attribute.Type.Float),
                 (Attribute.Type.Int, Attribute.Type.Float)}
        return (ta,tb) in casts

    ########################################

    def __init__(self, elem_type, elem_size:int=1, default_value=None):
        """
        __init__ method and the whole Attribute class are not supposed to be manipulated outside of the DataContainer class

        Parameters:
            elem_type : the type of the attribute (bool, int, float, complex, string)

            elem_size (int, optional): Number of elem_type objects to be stored per element. Defaults to 1.
            
            default_value (optional) : the default value of the attribute is n_elem is not specified. 
                If it is not specified either, it will correspond to the default value of the type provided in elem_type.
        """
        self.type : Attribute.Type = Attribute.Type(elem_type)
        self.elemsize : int = elem_size
        self._default_value = default_value
        self._check_default_value_type()
        self._data = None
    
    @property
    def default_value(self):
        if self._default_value is None:
            self._default_value = self.type.default_value(self.elemsize)
        return self._default_value

    def _check_default_value_type(self):
        if self._default_value is None : return
        if Attribute.Type(type(self._default_value)) != self.type:
            raise Attribute.DefaultValueTypeDoesNotMatchError(self._default_value, self.type)

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
    def __iter__(self):
        """
        Sparse attributes allow to iterate only over non-default elements
        """
        pass

    @abstractmethod
    def _expand(self, n : int):
        """Expands the storage capacity of the attributes. Adds `n` to self.n_elem
        Parameters:
            n (int) : number of new elements
        """
        pass

    @abstractmethod
    def as_array(self):
        pass

    def empty(self):
        """
        Check if an attribute is empty.
        For an attribute to be empty, its number of elements should not be fixed, and the dictionnary should be empty
        """
        return len(self)==0
    
    @abstractmethod
    def clear(self):
        """
        Empties the attribute. Frees the memory and ensures that all access return default value
        """
        pass


class Attribute(_BaseAttribute):
    def __init__(self, elem_type, elem_size:int=1, default_value=None):
        """
        __init__ method and the whole Attribute class are not supposed to be manipulated outside of the DataContainer class

        Parameters:
            elem_type : the type of the attribute (bool, int, float, complex, string)

            elem_size (int, optional): Number of elem_type objects to be stored per element. Defaults to 1.
            
            default_value (optional) : the default value of the attribute is n_elem is not specified. 
                If it is not specified either, it will correspond to the default value of the type provided in elem_type.
        """
        self.type : Attribute.Type = Attribute.Type(elem_type)
        self.elemsize : int = elem_size
        self._default_value = default_value
        self._check_default_value_type()
        self._data = dict()
    
    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]
        return self.default_value

    def __setitem__(self, key, value):
        if self.elemsize>1:
            data = list(value)
            n = len(data)
            if not n==self.elemsize: 
                # provided number of element do not match elemsize
                raise Attribute.InvalidSizeError(n,self.elemsize)
        
            datatype = type(data[0])
            data_attr_type = Attribute.Type(datatype)  
            if not self._can_be_casted(data_attr_type, self.type):
                raise Attribute.TypeNotMatchingError(data, datatype, self.type)
            self._data[key] = Vec(data)
        
        else:
            datatype = type(value)
            data_attr_type = Attribute.Type(datatype)
            if not self._can_be_casted(data_attr_type, self.type):
                raise Attribute.TypeNotMatchingError(value, datatype, self.type)
            self._data[key] = value

    def _expand(self, n : int):
        """Expands the storage capacity of the attributes. Adds `n` to self.n_elem
        Parameters:
            n (int) : number of new elements
        """
        pass # Nothing to do when data is stored in a dict

    def __len__(self):
        return self._data.__len__()

    def __iter__(self):
        """
        Sparse attributes allow to iterate only over non-default elements
        """
        return self._data.keys().__iter__()

    def as_array(self, container_size):
        out = np.full((container_size, self.elemsize), self.default_value, dtype= self.type.dtype)
        for i,x in self._data.items():
            out[i,:] = x
        return np.squeeze(out)

    def clear(self):
        """
        Empties the attribute. Frees the memory and ensures that all access return default value
        """
        self._data = dict()

class ArrayAttribute(_BaseAttribute):
    def __init__(self, elem_type, n_elem:int, elem_size:int=1, default_value=None):
        """
        __init__ method and the whole Attribute class are not supposed to be manipulated outside of the DataContainer class.
        An ArrayAttribute stores its values in a numpy array. This is a less flexible but safer approach than Attribute.

        Parameters:
            elem_type : the type of the attribute (bool, int, float, complex, string)

            n_elem (int): Total number of elements in the container. Should match the size of the DataContainer the attribute is stored in.

            elem_size (int, optional): Number of elem_type objects to be stored per element. Defaults to 1.
            
            dense (bool, optional): _description_. Defaults to False.

            default_value (optional) : the default value of the attribute is n_elem is not specified. 
                If it is not specified either, it will correspond to the default value of the type provided in elem_type.
        """
        self.type : Attribute.Type = Attribute.Type(elem_type)
        self.elemsize : int = elem_size
        self.n_elem:int = n_elem
        self._default_value = default_value
        self._check_default_value_type()
        self._data = np.full((n_elem, elem_size), self.default_value, dtype= self.type.dtype)
    
    def _check_out_of_bounds(self,key):
        if key<0 or key>self.n_elem:
            raise Attribute.OutOfBoundsError(key, self.n_elem)

    def __getitem__(self, key):
        self._check_out_of_bounds(key)
        return self._data[key,0] if self.elemsize==1 else self._data[key,:]

    def __setitem__(self, key, value):
        self._check_out_of_bounds(key)
        if self.elemsize>1:
            data = list(value)
            n = len(data)
            if not n==self.elemsize: 
                # provided number of element do not match elemsize
                raise Attribute.InvalidSizeError(n,self.elemsize)
        
            datatype = type(data[0])
            data_attr_type = Attribute.Type(datatype)  
            if not self._can_be_casted(data_attr_type, self.type):
                raise Attribute.TypeNotMatchingError(data, datatype, self.type)
            self._data[key] = Vec(data)
        
        else:
            datatype = type(value)
            data_attr_type = Attribute.Type(datatype)
            if not self._can_be_casted(data_attr_type, self.type):
                raise Attribute.TypeNotMatchingError(value, datatype, self.type)
            self._data[key] = value

    def _expand(self, n : int):
        """Expands the storage capacity of the attributes. Adds `n` to self.n_elem
        Parameters:
            n (int) : number of new elements
        """
        self._data = np.concatenate((self._data, np.full((n, self.elemsize), self.default_value, dtype= self.type.dtype)))
        self.n_elem += n

    def __len__(self):
        return self.n_elem

    def __iter__(self):
        """
        Sparse attributes allow to iterate only over non-default elements
        """
        return range(self.n_elem)

    def as_array(self, *args):
        return np.squeeze(self._data)
    
    def clear(self):
        """
        Empties the attribute. Frees the memory and ensures that all access return default value
        """
        self._data = np.full((self.n_elem, self.elemsize), self.default_value, dtype= self.type.dtype)