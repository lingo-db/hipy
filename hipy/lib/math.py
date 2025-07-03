__HIPY_MODULE__ = "math"

import hipy
import sys
import hipy.intrinsics as intrinsics

hipy.register(sys.modules[__name__])

@hipy.compiled_function
def fact(l):
    r =1
    for i in range(1, l):
        r *= (i+1)
    return r
