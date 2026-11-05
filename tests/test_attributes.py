import pytest
import numpy as np
import mouette as M
from mouette.mesh.mesh_attributes import AttributeType, _BaseAttribute, Attribute, SparseAttribute
from mouette.mesh.mesh_data import DataContainer
from utils import *

dense_and_sparse = pytest.mark.parametrize("sparse", [True, False])
Err = _BaseAttribute

@pytest.fixture
def test_container():
    data = np.array([[0,1,2], [4,5,6], [0,3,4]])
    container = DataContainer(data, id="test")
    return container

########## Container ##########

def test_container_size(test_container):
    assert test_container.size == 3
    assert len(test_container) == 3

def test_container_attributes_listing(test_container):
    test_container.create_attribute("a", float)
    test_container.create_attribute("b", int, sparse=True)
    assert set(test_container.attributes) == {"a", "b"}
    assert isinstance(test_container.get_attribute("a"), Attribute)
    assert isinstance(test_container.get_attribute("b"), SparseAttribute)
    assert not test_container.get_attribute("a").is_sparse
    assert test_container.get_attribute("b").is_sparse

@dense_and_sparse
def test_attribute_has_get_delete(test_container, sparse):
    assert not test_container.has_attribute("test_attr")
    test_container.create_attribute("test_attr", float, 1, sparse=sparse)
    assert test_container.has_attribute("test_attr")
    attr = test_container.get_attribute("test_attr")
    attr.clear()
    test_container.delete_attribute("test_attr")
    assert not test_container.has_attribute("test_attr")

def test_container_get_missing(test_container):
    with pytest.raises(KeyError):
        test_container.get_attribute("nope")

def test_container_delete_missing(test_container):
    test_container.delete_attribute("nope") # nothing should happen

def test_container_duplicate_override(test_container, monkeypatch):
    monkeypatch.setattr(M.config, "display_duplicate_attribute_warning", False)
    a = test_container.create_attribute("a", float)
    a[0] = 1.
    b = test_container.create_attribute("a", int)
    assert b.type == AttributeType.Int
    assert b[0] == 0

def test_container_duplicate_warning(test_container, monkeypatch):
    monkeypatch.setattr(M.config, "display_duplicate_attribute_warning", True)
    a = test_container.create_attribute("a", float)
    with pytest.warns(UserWarning):
        b = test_container.create_attribute("a", int)

@pytest.mark.parametrize("shape", [(3,), (3,1), (3,4)])
def test_register_array(test_container, shape):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    attr = test_container.register_array_as_attribute("r", data)
    assert attr.type == AttributeType.Float
    assert attr.n_elem == 3
    if len(shape)==1:
        assert attr.elem_size==1
    else:
        assert attr.elem_size==shape[1]
    assert np.allclose(np.asarray(attr).reshape(shape), data)

def test_register_array_wrong_shape(test_container):
    with pytest.raises(ValueError):
        test_container.register_array_as_attribute("r", np.zeros(5))

@dense_and_sparse
def test_corner_container_attribute(sparse):
    c = M.mesh.mesh_data.CornerDataContainer(np.array([0,1,2,0]), np.array([0,0,0,1]), id="corners")
    attr = c.create_attribute("a", float, 1, sparse=sparse)
    assert attr.n_elem == 4
    attr[3] = 2.
    assert attr[3] == 2.

########## Sizes ##########

@dense_and_sparse
def test_attribute_n_elem(test_container, sparse):
    attr = test_container.create_attribute("test_attr", float, 2, sparse=sparse)
    assert len(attr) == attr.n_elem == len(test_container)

@dense_and_sparse
@pytest.mark.parametrize("bad", [[1.], [1., 2., 3.]])
def test_attribute_vec_wrong_size(test_container, sparse, bad):
    attr = test_container.create_attribute("a", float, 2, sparse=sparse)
    with pytest.raises(Err.InvalidSizeError):
        attr[0] = bad

@dense_and_sparse
def test_attribute_vec_accepts_ndarray_and_tuple(test_container, sparse):
    attr = test_container.create_attribute("a", float, 3, sparse=sparse)
    attr[0] = np.array([1., 2., 3.])
    attr[1] = (4., 5., 6.)
    assert np.allclose(attr[0], [1,2,3])
    assert np.allclose(attr[1], [4,5,6])

@dense_and_sparse
@pytest.mark.parametrize("elem_size", [1, 3])
def test_attribute_single_element_container(sparse, elem_size): # [spec] np.squeeze edge case
    c = DataContainer(np.array([[0,1,2]]), id="one")
    attr = c.create_attribute("a", float, elem_size, sparse=sparse)
    val = 1. if elem_size==1 else [1., 2., 3.]
    attr[0] = val
    assert np.allclose(attr[0], val)
    if elem_size > 1:
        assert np.asarray(attr).shape == (1, elem_size)

@dense_and_sparse
def test_attribute_empty_container(sparse):
    c = M.mesh.mesh_data.DataContainer(np.zeros((0,3), dtype=int), id="empty")
    attr = c.create_attribute("a", float, 1, sparse=sparse)
    assert attr.n_elem == 0
    with pytest.raises(IndexError):
        attr[0] = 1.

########## Get / Set ##########

@dense_and_sparse
def test_attribute_set_get(test_container, sparse):
    attr = test_container.create_attribute("test_attr", float, 1, sparse=sparse)
    attr[0] = 42.
    attr[1] = 12 # cast an int into a float
    assert attr[0] == 42.
    assert attr[1] == 12.
    assert attr[2] == 0.

@dense_and_sparse
def test_attribute_out_of_bounds(test_container, sparse):
    attr = test_container.create_attribute("test_attr", float, 1, sparse=sparse)
    with pytest.raises(IndexError):
        attr[4] = 42.
    with pytest.raises(IndexError):
        attr[-1] = 42.

@dense_and_sparse
def test_attribute_getitem_out_of_bounds(test_container, sparse):
    attr = test_container.create_attribute("a", float, 1, sparse=sparse)
    with pytest.raises(IndexError):
        attr[3]
    with pytest.raises(IndexError):
        attr[-1]

@dense_and_sparse
def test_attribute_numpy_int_key(test_container, sparse):
    attr = test_container.create_attribute("a", float, 1, sparse=sparse)
    attr[np.int64(1)] = 3.
    assert attr[1] == 3.

@dense_and_sparse
def test_attribute_vec_element_is_mutable(test_container, sparse):
    attr = test_container.create_attribute("a", float, 2, sparse=sparse)
    attr[0] = [23.52, 35.12]
    attr[0][0] = 11.111
    assert (attr[0] == [11.111, 35.12]).all()

########## Types and casting ##########

@dense_and_sparse
def test_attribute_types(test_container, sparse):
    attr_bool    = test_container.create_attribute("test_bool",    bool,    1, sparse=sparse)
    attr_int     = test_container.create_attribute("test_int",     int,     1, sparse=sparse)
    attr_float   = test_container.create_attribute("test_float",   float,   1, sparse=sparse)
    attr_complex = test_container.create_attribute("test_complex", complex, 1, sparse=sparse)

    attr_bool[0] = True
    assert attr_bool[0] == True

    attr_int[0] = 42
    assert attr_int[0] == 42

    attr_float[0] = 23.52
    assert attr_float[0] == 23.52

    attr_complex[0] = 10+22.5j
    assert attr_complex[0] == 10+22.5j

@dense_and_sparse
def test_attribute_types_vec(test_container, sparse):
    attr_bool    = test_container.create_attribute("test_bool",    bool,    2, sparse=sparse)
    attr_int     = test_container.create_attribute("test_int",     int,     2, sparse=sparse)
    attr_float   = test_container.create_attribute("test_float",   float,   2, sparse=sparse)
    attr_complex = test_container.create_attribute("test_complex", complex, 2, sparse=sparse)

    attr_bool[0] = [True, False]
    assert (attr_bool[0] == [True, False]).all()

    attr_int[0] = [42, 10]
    assert (attr_int[0] == [42,10]).all()

    attr_float[0] = [23.52, 35.12]
    assert (attr_float[0] == [23.52, 35.12]).all()

    attr_complex[0] = [10+22.5j, 9+9j]
    assert (attr_complex[0] == [10+22.5j, 9+9j]).all()

@dense_and_sparse
def test_attribute_cast(test_container, sparse):
    attr = test_container.create_attribute("a", float, 1, sparse=sparse)
    attr[0] = True
    assert attr[0] == 1.
    attr[0] = 1
    assert attr[0] == 1.
    with pytest.raises(Err.TypeNotMatchingError):
        attr[0] = 1+2j

@dense_and_sparse
def test_attribute_int_rejects_float(test_container, sparse):
    attr = test_container.create_attribute("a", int, 1, sparse=sparse)
    with pytest.raises(Err.TypeNotMatchingError):
        attr[0] = 1.5

@dense_and_sparse
def test_attribute_vec_type_mismatch(test_container, sparse):
    attr = test_container.create_attribute("a", int, 2, sparse=sparse)
    with pytest.raises(Err.TypeNotMatchingError):
        attr[0] = [1.5, 2.5]

@dense_and_sparse
@pytest.mark.parametrize("val", [np.float32(1.5), np.float64(1.5), np.int64(3), np.bool_(True)])
def test_attribute_numpy_scalars(test_container, sparse, val):
    attr = test_container.create_attribute("a", float, 1, sparse=sparse)
    attr[0] = val
    assert attr[0] == float(val)

@dense_and_sparse
@pytest.mark.parametrize("typ", [np.float32, np.float64, np.int32, np.int64, np.bool_, np.complex128])
def test_attribute_numpy_types(test_container, sparse, typ):
    attr = test_container.create_attribute("a", typ, 1, sparse=sparse)
    assert attr[1] == attr.type.default_value()

def test_attribute_invalid_type(test_container):
    with pytest.raises(ValueError):
        test_container.create_attribute("a", list, 1)

def test_type_helpers():
    T = AttributeType
    assert T.from_string("double") == T.Float
    assert T.from_string('"index_t"') == T.Int
    assert T.from_string("bool") == T.Bool
    assert T.from_string("complex") == T.Complex
    with pytest.raises(Exception):
        T.from_string("unknown_type")
    assert T.Float.to_string() == "double"
    assert T.Int.to_string() == "int"
    assert T.Float.byte_size() == 8
    assert T.can_be_casted(T.Int, T.Float)
    assert T.can_be_casted(T.Bool, T.Int)
    assert not T.can_be_casted(T.Float, T.Int)
    assert not T.can_be_casted(T.Complex, T.Float)

########## Default values ##########

@dense_and_sparse
def test_attribute_default_value(test_container, sparse):
    attr = test_container.create_attribute("test_attr", float, 1, sparse=sparse, default_value=42.)
    attr[1] = 76.5
    assert attr[0] == 42.
    assert attr[1] == 76.5
    assert attr[2] == 42.


@dense_and_sparse
def test_attribute_default_value_wrong_type(test_container, sparse):
    with pytest.raises(Err.DefaultValueTypeDoesNotMatchError):
        test_container.create_attribute("a", float, 1, sparse=sparse, default_value="x")

def test_sparse_default_not_shared(test_container): 
    # [spec] mutating an unset element must not alter the default
    attr = test_container.create_attribute("a", float, 2, sparse=True)
    v = attr[0]
    v[0] = 99.
    assert attr[1][0] == 0.

########## Delete / clear ##########

@dense_and_sparse
def test_attribute_delitem(test_container, sparse):
    attr = test_container.create_attribute("a", float, 1, sparse=sparse, default_value=7.)
    attr[0] = 1.
    del attr[0]
    assert attr[0] == 7.

@dense_and_sparse
def test_attribute_clear_resets(test_container, sparse):
    attr = test_container.create_attribute("a", float, 2, sparse=sparse)
    attr[0] = [1., 2.]
    attr[2] = [3., 4.]
    attr.clear()
    assert np.allclose(np.asarray(attr), 0.)
    attr[1] = [5., 6.] # still writable after clear
    assert np.allclose(attr[1], [5., 6.])

########## Iteration / array conversion ##########

def test_dense_iter_yields_values(test_container):
    attr = test_container.create_attribute("a", int, 1)
    for i in range(3):
        attr[i] = i*10
    assert list(attr) == [0, 10, 20]

def test_sparse_iter_yields_keys(test_container):
    attr = test_container.create_attribute("a", int, 1, sparse=True)
    attr[2] = 5
    attr[0] = 1
    assert sorted(attr) == [0, 2]

@dense_and_sparse
def test_attribute_as_array(test_container, sparse):
    attr = test_container.create_attribute("test_attr", float, 1, sparse=sparse)
    attr[0] = 42.
    attr[1] = 76.52
    attr[2] = -26.5
    assert np.allclose(np.asarray(attr), np.array([42., 76.52, -26.5]))

@dense_and_sparse
def test_attribute_as_array_vec(test_container, sparse):
    attr = test_container.create_attribute("a", float, 2, sparse=sparse)
    attr[1] = [1., 2.]
    arr = np.asarray(attr)
    assert arr.shape == (3,2)
    assert np.allclose(arr, [[0,0], [1,2], [0,0]])
    assert attr.as_array(dtype=np.float32).dtype == np.float32

@dense_and_sparse
def test_attribute_as_array_copy(test_container, sparse):
    attr = test_container.create_attribute("a", float, 1, sparse=sparse)
    arr = attr.as_array(copy=True)
    arr[0] = 5.
    assert attr[0] == 0.
