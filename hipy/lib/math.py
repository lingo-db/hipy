__HIPY_MODULE__ = "math"

import hipy
import sys
import hipy.intrinsics as intrinsics
import math as pymath
hipy.register(sys.modules[__name__])

@hipy.compiled_function
def fact(l):
    r =1
    for i in range(1, l):
        r *= (i+1)
    return r

@hipy.compiled_function
def ceil(x):
    return intrinsics.call_builtin("scalar.float.ceil",int,  [x])

@hipy.compiled_function
def exp(x):
    return intrinsics.call_builtin("scalar.float.exp",float,  [x])

@hipy.compiled_function
def sqrt(x):
    return intrinsics.call_builtin("scalar.float.sqrt",float,  [x])

@hipy.compiled_function
def log(x):
    return intrinsics.call_builtin("scalar.float.log",float,  [x])

@hipy.compiled_function
def sin(x):
    return intrinsics.call_builtin("scalar.float.sin",float,  [x])

@hipy.compiled_function
def cos(x):
    return intrinsics.call_builtin("scalar.float.cos",float,  [x])

@hipy.compiled_function
def acos(x):
    return intrinsics.call_builtin("scalar.float.acos",float,  [x])

@hipy.compiled_function
def atan2(x, y):
    return intrinsics.call_builtin("scalar.float.atan2",float,  [x, y])
pi = pymath.pi