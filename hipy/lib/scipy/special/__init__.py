__HIPY_MODULE__ = "scipy.special"
import numpy as np
import hipy.lib.numpy
import sys

import hipy
from hipy import intrinsics, ir
from hipy.value import SimpleType, Value, Type, raw_module

hipy.register(sys.modules[__name__])

@hipy.compiled_function
def erf(x):
    return np._array_apply_scalar(x, lambda v: intrinsics.call_builtin("scalar.float.erf", np.float64,[v]))
