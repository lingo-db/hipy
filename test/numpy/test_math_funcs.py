import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np

@hipy.compiled_function
def fn_sin():
    print(np.sin(0.0))
    print(np.sin(1.0))
    print(np.sin(2))
    print(np.sin(np.ones(3)))

def test_sin():
    check_prints(fn_sin, """
0.0
0.841471
0.909297
[0.84147098 0.84147098 0.84147098]
""")