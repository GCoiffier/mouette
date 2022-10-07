"""
mouette.utils.iterators
Various iterators on list to make life easier
"""

def cyclic_pairs(L: list):
    """pairs of form (L[i], L[i+1]) that wraps at the end

    Parameters:
        L (list): list
    """
    n = len(L)
    for i in range(n):
        yield L[i],L[(i+1)%n]

def cyclic_pairs_enumerate(L : list):
    """pairs of form ((i,i+1),(L[i], L[i+1]) that wraps at the end

    Parameters:
        L (list): list
    """
    n = len(L)
    for i in range(n):
        yield (i, (i+1)%n), (L[i],L[(i+1)%n])


def cyclic_triplets(L : list):
    """
    triplets of form (L[i-1], L[i], L[i+1]) that wraps at the end

    Args:
        L (list): list
    """
    n = len(L)
    for i in range(n):
        yield L[i-1], L[i], L[(i+1)%n]

def consecutive_pairs(L : list):
    """pairs of form (L[i], L[i+1]) that does not wrap at the end

    Parameters:
        L (list): list
    """
    for i in range(len(L)-1):
        yield L[i],L[i+1]

def consecutive_triplets(L : list):
    """
    triplets of form (L[i-1], L[i], L[i+1]) that does not wraps at the end

    Args:
        L (list): list
    """
    n = len(L)
    for i in range(1, len(L)-1):
        yield L[i-1], L[i], L[i+1]

def cyclic_permutations(L):
    """All the cyclic permutations of elements of L

    Parameters:
        L (list): list
    """
    n = len(L)
    for j in range(n):
        yield [L[i - j] for i in range(n)]

def cyclic_perm_enumerate(L):
    """All the cyclic permutations of elements of L with the index of the first element.

    Parameters:
        L (list): list
    """
    n = len(L)
    for j in range(n):
        yield [((i-j)%n , L[i - j]) for i in range(n)]

def offset(L, k :int):
    n = len(L)
    return [L[(i+k)%n] for i in range(n)]