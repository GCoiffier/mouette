import pytest
import mouette as M

########## iterators ##########

def consecutive_pairs():
    L = [1,2,3,4,5]
    pairs = list(M.utils.consecutive_pairs_pairs(L))
    assert pairs == [(1,2), (2,3), (3,4), (4,5)]

def test_cyclic_pairs():
    L = [1,2,3,4,5]
    pairs = list(M.utils.cyclic_pairs(L))
    assert pairs == [(1,2), (2,3), (3,4), (4,5), (5,1)]

def test_cyclic_pairs_enumerate():
    L = [1,2,3,4,5]
    pairs = list(M.utils.cyclic_pairs_enumerate(L))
    assert pairs == [((0,1),(1,2)), ((1,2),(2,3)), ((2,3),(3,4)), ((3,4),(4,5)), ((4,0),(5,1))]

def test_consecutive_triplets():
    L = [1,2,3,4,5]
    triplets = list(M.utils.consecutive_triplets(L))
    assert triplets == [(1,2,3), (2,3,4), (3,4,5)]

def test_cyclic_triplets():
    L = [1,2,3,4,5]
    triplets = list(M.utils.cyclic_triplets(L))
    assert triplets == [(5,1,2), (1,2,3), (2,3,4), (3,4,5), (4,5,1)]

def test_cyclic_permutations():
    L = [1,2,3,4]
    perms = [x for x in M.utils.cyclic_permutations(L)]
    assert perms == [
        [1,2,3,4],
        [4,1,2,3],
        [3,4,1,2],
        [2,3,4,1],
    ]

def test_cyclic_perm_enumerate():
    L = [1,2,3,4]
    perms = [x for x in M.utils.cyclic_perm_enumerate(L)]
    assert perms == [
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        [(3, 4), (0, 1), (1, 2), (2, 3)],
        [(2, 3), (3, 4), (0, 1), (1, 2)],
        [(1, 2), (2, 3), (3, 4), (0, 1)],
    ]

def test_offset():
    L = [1,2,3,4,5]
    assert M.utils.offset(L,2)==[3,4,5,1,2]

########## misc ##########

def test_filename():
    assert M.utils.get_filename("path/to/test_file.txt") == "test_file"
    assert M.utils.get_filename("path/to/foo.obj") == "foo"
    assert M.utils.get_filename("path/to/mysuperfile.geogram_ascii") == "mysuperfile"

def test_extension():
    assert M.utils.get_extension("path/to/test_file.txt") == "txt"
    assert M.utils.get_extension("path/to/foo.obj") == "obj"
    assert M.utils.get_extension("path/to/mysuperfile.geogram_ascii") == "geogram_ascii"

def test_replace_extension():
    assert M.utils.replace_extension("path/to/file.txt", ".obj") == "path/to/file.obj"
    assert M.utils.replace_extension("path/to/foo.meh", ".txt") == "path/to/foo.txt"

def test_keyify():
    assert M.utils.keyify(3,2) == (2,3)
    assert M.utils.keyify([1,3,2]) == (1,2,3)
    assert M.utils.keyify((4,5)) == (4,5)

def test_replace_in_list():
    assert M.utils.replace_in_list([1,2,3],2,42) == [1,42,3]
    assert M.utils.replace_in_list(["a","b","a","c"], "a", 16) == [16,"b",16,"c"]

def test_logger():
    log = M.utils.Logger("test_logger", True)
    log.log("this is a test")
    assert True

########## Priority queue ##########

def test_priority_queue_empty():
    p = M.utils.PriorityQueue()
    assert p.empty()

def test_priority_queue_push_pop():
    p = M.utils.PriorityQueue()
    p.push("test",0)
    assert p.front.x == "test"
    assert p.pop().x == "test"
    assert p.empty()

def test_priority_pop_from_empty():
    p = M.utils.PriorityQueue()
    try:
        _ = p.get()
        assert False
    except IndexError:
        assert True

def test_priority_sort():
    p = M.utils.PriorityQueue()
    p.push("a",5)
    p.push("b",2)
    elem = p.get()
    assert elem.x == "b"
    assert elem.priority == 2

########## Union Find ##########

def test_UF_create():
    uf = M.utils.UnionFind()
    assert len(uf)==0
    uf = M.utils.UnionFind(range(3))
    assert len(uf)==3

def test_UF_add():
    uf = M.utils.UnionFind()
    for i in range(3):
        uf.add(i)
    assert len(uf)==3
    assert 1 in uf

def test_UF_get_set():
    uf = M.utils.UnionFind([0,1])
    assert uf[0]==0
    assert uf[1]==1
    uf[0] = 3
    assert uf[0]==3
    assert uf[1]==1

def test_UF_find():
    uf = M.utils.UnionFind(range(3))
    assert uf.find(0)==0
    assert uf.find(1)==1
    assert uf.find(2)==2
    try:
        uf.find(3)
        assert False
    except ValueError:
        assert True

def test_UF_union():
    uf = M.utils.UnionFind(range(3))
    assert not uf.connected(0,1)
    uf.union(0,1)
    assert uf.connected(0,1)
    uf.union(3,0) # should automatically add element 3
    assert uf.connected(0,3)
    assert len(uf)==4

def test_UF_component():
    uf = M.utils.UnionFind(range(3))
    uf.union(0,1)
    assert uf.component(0)==uf.component(1)
    assert uf.component(0)=={0,1}
    try:
        uf.component(42)
        assert True
    except ValueError:
        assert True
    assert uf.components() == [[0,1], [2]]

def test_UF_roots():
    uf = M.utils.UnionFind(range(3))
    uf.union(0,1)
    assert 0 in uf.roots() or 1 in uf.roots()
    assert 2 in uf.roots()

def test_UF_component_mapping():
    uf = M.utils.UnionFind(range(3))
    uf.union(0,1)
    m = uf.component_mapping()
    assert len(m[0])==2
    assert len(m[2])==1

def test_UF_repr():
    uf = M.utils.UnionFind(range(3))
    print(uf)
    assert True