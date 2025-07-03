import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np

@hipy.compiled_function
def fn_scalar_ops():
    arr = np.ones(3, dtype=np.int64)
    print(arr*2)
    print(arr+2)
    print(2*arr)
    print(2+arr)

def test_scalar_ops():
    check_prints(fn_scalar_ops, """
[2 2 2]
[3 3 3]
[2 2 2]
[3 3 3]
""")

@hipy.compiled_function
def fn_binary_ops():
    arr = np.ones(3, dtype=np.int64)
    print(arr*arr)
    print(arr+arr)

def test_binary_ops():
    check_prints(fn_binary_ops, """
[1 1 1]
[2 2 2]
""")