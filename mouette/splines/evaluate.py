from ..utils.argument_check import InvalidRangeArgumentError

def de_casteljau(P:list, t:float):
    """De Casteljau's algorithm for evaluating a Bezier curve B_P at some value of the parameter t

    Args:
        P (list): Control points of the Bezier curve
        t (float): parameter value

    Raises:
        InvalidRangeArgumentError: if t is not in the [0;1] interval

    Returns:
        Vec: the position in space of B_P(t) 
    """
    if not 0 <= t <= 1:
        raise InvalidRangeArgumentError("t", t, "in [0,1]")
    coeffs = [x for x in P]
    order = len(P)-1
    for j in range(order):
        for i in range(order - j):
            coeffs[i] = t*coeffs[i+1] + (1-t)*coeffs[i]
    return coeffs[0]