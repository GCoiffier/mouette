"""
argument_check.py

Utility functions and exceptions for sanity check on function arguments
"""

class InvalidArgumentTypeError(Exception):
    def __init__(self, argname :str, argtype : type, *expected_types:type):
        message = f"Argument '{argname}' has type {argtype} while the function expected types {expected_types}"
        super().__init__(message)

class InvalidArgumentValueError(Exception):
    def __init__(self, argname :str, argvalue : type, expected_values:list = None):
        if expected_values is None:
            message = f"Argument '{argname}' has invalid value '{argvalue}'"
        else:
            message = f"Argument '{argname}' has invalid value {argvalue}. Allowed values are {expected_values}"
        super().__init__(message)

class InvalidRangeArgumentError(Exception):
    def __init__(self, argname:str, argvalue, condition:str):
        message = f"Argument {argname} is out of range. Expected a value {condition}. Got {argvalue}"
        super().__init__(message)

class NegativeArgumentError(Exception):
    def __init__(self, argname:str, argvalue):
        message = f"Argument '{argname}' has value {argvalue} but should be >=0"
        super().__init__(message)

def check_argument(argname :str, argvalue, expected_type: type, expected_values:list=None):
    """
    check input sanity on the argument of a function

    Args:
        argname (str): Name of the argument (for exhaustivity of error message)
        argvalue (any): value of the argument
        
        expected_type (type): expected type of the argument. 
            If type(argvalue) does not match, will raise an InvalidArgumentTypeError

        expected_values (list, optional): List of possible values the argument is allowed to take. 
            If argvalue is not in the list, will raise an InvalidArgumentValueError.
            Defaults to None.
    """
    if not isinstance(argvalue, expected_type):
        raise InvalidArgumentTypeError(argname, type(argvalue), expected_type)
    if expected_values is not None and argvalue not in expected_values:
        raise InvalidArgumentValueError(argname, argvalue, expected_values)