"""
mouette.utils.iterators
Various iterators on list to make life easier
"""

def cyclic_pairs(L: list):
    """pairs of form (L[i], L[i+1]) that wraps at the end

    Args:
        L (list): list
    """
    n = len(L)
    for i in range(n):
        yield L[i],L[(i+1)%n]


def consecutive_pairs(L : list):
    """pairs of form (L[i], L[i+1]) that does not wrap at the end

    Args:
        L (list): list
    """
    for i in range(len(L)-1):
        yield L[i],L[i+1]

def cyclic_permutations(L):
    """All the cyclic permutations of elements of L

    Args:
        L (list): list
    """
    n = len(L)
    for j in range(n):
        yield [L[i - j] for i in range(n)]

def cyclic_perm_enumerate(L):
    """All the cyclic permutations of elements of L with the index of the first element.

    Args:
        L (list): list
    """
    n = len(L)
    for j in range(n):
        yield [((i-j)%n , L[i - j]) for i in range(n)]

def offset(L, k :int):
    n = len(L)
    return [L[(i+k)%n] for i in range(n)]