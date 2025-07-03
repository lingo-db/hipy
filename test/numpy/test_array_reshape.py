import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np



@hipy.compiled_function
def fn_basic():
    arr= np.ones(9, dtype=np.int64)
    print(arr.reshape((3,3)))
    arr = np.ones((3,3), dtype=np.int64)
    print(arr.reshape(9))

def test_basic():
    check_prints(fn_basic, """
[[1 1 1]
 [1 1 1]
 [1 1 1]]
[1 1 1 1 1 1 1 1 1]
""")

@hipy.compiled_function
def fn_view():
    arr = np.ones(9, dtype=np.int64)
    print(arr[1:].reshape((2,4)))
    arr = np.ones((2,4), dtype=np.int64)
    print(arr[:,1:-1].reshape((4,1)))

def test_view():
    check_prints(fn_view, """
[[1 1 1 1]
 [1 1 1 1]]
[[1]
 [1]
 [1]
 [1]]
""")

