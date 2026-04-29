import pytest
import mouette as M
from utils import *

@pytest.fixture
def test_container():
    container = M.mesh.mesh_data.DataContainer(id="test")
    container.append([0,1,2])
    container.append([4,5,6])
    container.append([0,3,4])
    return container

def test_attribute_sparse(test_container):
    attr = test_container.create_attribute("test_attr",float,1)
    attr[0] = 42.
    assert attr[0] == 42.
    assert attr[1] == 0.

def test_attribute_dense(test_container):
    attr = test_container.create_attribute("test_attr",float,1, dense=True)
    attr[0] = 42.
    assert attr[0] == 42.
    assert attr[1] == 0.
    assert attr[2] == 0.

def test_attribute_as_array(test_container):
    attr = test_container.create_attribute("test_attr",float,1)
    attr[0] = 42.
    attr[1] = 76.52
    attr[2] = -26.5
    diff = abs(attr.as_array(len(test_container)) - np.array([42., 76.52, -26.5]))
    assert np.all(diff < 1e-8)

def test_attribute_sparse_cast(test_container):
    attr = test_container.create_attribute("test_attr",float,1)
    attr[0] = True
    assert attr[0] == 1.
    attr[0] = 1
    assert attr[0] == 1.

    try:
        attr[0] = "toto"
        assert False
    except:
        assert True
    
    try:
        attr[0] = 1+2j
        assert False
    except:
        assert True

def test_attribute_has_get_delete(test_container):
    assert not test_container.has_attribute("test_attr")
    test_container.create_attribute("test_attr",float,1)
    assert test_container.has_attribute("test_attr")
    attr = test_container.get_attribute("test_attr")
    attr.clear()
    test_container.delete_attribute("test_attr")
    assert not test_container.has_attribute("test_attr")

def test_attribute_default_value(test_container):

    attr = test_container.create_attribute("test_attr",float,1, default_value=42.)
    attr[1] = 76.5
    assert attr[0] == 42.
    assert attr[1] == 76.5
    assert attr[2] == 42.

def test_attribute_types(test_container):
    attr_bool    = test_container.create_attribute("test_bool",bool,1)
    attr_int     = test_container.create_attribute("test_int",int,1)
    attr_float   = test_container.create_attribute("test_float",float,1)
    attr_complex = test_container.create_attribute("test_complex",complex,1)
    attr_str     = test_container.create_attribute("test_str",str,1)

    attr_bool[0] = True
    assert attr_bool[0] == True

    attr_int[0] = 42
    assert attr_int[0] == 42
    
    attr_float[0] = 23.52
    assert attr_float[0] == 23.52

    attr_complex[0] = 10+22.5j
    assert attr_complex[0] == 10+22.5j 

    attr_str[0] = "bonjour"
    assert attr_str[0] == "bonjour"


def test_attribute_types_vec(test_container):
    attr_bool    = test_container.create_attribute("test_bool",bool,2)
    attr_int     = test_container.create_attribute("test_int",int,2)
    attr_float   = test_container.create_attribute("test_float",float,2)
    attr_complex = test_container.create_attribute("test_complex",complex,2)
    attr_str     = test_container.create_attribute("test_str",str,2)

    attr_bool[0] = [True, False]
    assert (attr_bool[0] == [True, False]).all()

    attr_int[0] = [42, 10]
    assert (attr_int[0] == [42,10]).all()
    
    attr_float[0] = [23.52, 35.12]
    assert (attr_float[0] == [23.52, 35.12]).all()
    attr_float[0][0] = 11.111
    assert (attr_float[0] == [11.111, 35.12]).all()

    attr_complex[0] = [10+22.5j, 9+9j]
    assert (attr_complex[0] == [10+22.5j, 9+9j] ).all()

    attr_str[0] = ["bonjour", "merci"]
    assert (attr_str[0] == ["bonjour", "merci"]).all()

def test_attribute_out_of_bounds(test_container):
    attr = test_container.create_attribute("test_attr", float, 1, dense=True)
    try:
        attr[4] = 42.
        assert False
    except M.mesh.mesh_attributes.Attribute.OutOfBoundsError:
        assert True

    try:
        attr[-1] = 42.
        assert False
    except M.mesh.mesh_attributes.Attribute.OutOfBoundsError:
        assert True

def test_attribute_types_dense(test_container):
    attr_bool    = test_container.create_attribute("test_bool",    bool,    1, dense=True)
    attr_int     = test_container.create_attribute("test_int",     int,     1, dense=True)
    attr_float   = test_container.create_attribute("test_float",   float,   1, dense=True)
    attr_complex = test_container.create_attribute("test_complex", complex, 1, dense=True)
    attr_str     = test_container.create_attribute("test_str",     str,     1, dense=True)

    attr_bool[0] = True
    assert attr_bool[0] == True

    attr_int[0] = 42
    assert attr_int[0] == 42
    
    attr_float[0] = 23.52
    assert attr_float[0] == 23.52

    attr_complex[0] = 10+22.5j
    assert attr_complex[0] == 10+22.5j

    attr_str[0] = 'bonjour'
    assert attr_str[0] == 'bonjour'


def test_attribute_types_vec_dense(test_container):
    attr_bool    = test_container.create_attribute("test_bool",    bool,    2, dense=True)
    attr_int     = test_container.create_attribute("test_int",     int,     2, dense=True)
    attr_float   = test_container.create_attribute("test_float",   float,   2, dense=True)
    attr_complex = test_container.create_attribute("test_complex", complex, 2, dense=True)
    attr_str     = test_container.create_attribute("test_str",     str,     2, dense=True)

    attr_bool[0] = [True, False]
    assert (attr_bool[0] == [True, False]).all()

    attr_int[0] = [42, 10]
    assert (attr_int[0] == [42,10]).all()
    
    attr_float[0] = [23.52, 35.12]
    assert (attr_float[0] == [23.52, 35.12]).all()

    attr_float[0][0] = 11.111
    assert (attr_float[0] == [11.111, 35.12]).all()

    attr_complex[0] = [10+22.5j, 9+9j]
    assert (attr_complex[0] == [10+22.5j, 9+9j] ).all()

    attr_str[0] = ["bonjour", "merci"]
    assert (attr_str[0] == ["bonjour", "merci"]).all()