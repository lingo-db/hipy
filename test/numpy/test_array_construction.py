import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
from hipy.lib.numpy import _concrete_ndarray
import numpy as np



@hipy.compiled_function
def fn_empty_(shape):
    print(len(str(np.empty(shape,dtype= np.int64)).split("]")))

    print(len(str(np.empty(shape,dtype= np.float64)).split("]")))

    print(len(str(np.empty(shape)).split("]")))

@hipy.compiled_function
def fn_empty():
    fn_empty_(3)
    fn_empty_((3,))
    fn_empty_((3,3))

def test_empty():
    check_prints(fn_empty, """
2
2
2
2
2
2
5
5
5
""")


@hipy.compiled_function
def fn_ones_(shape):
    print(np.ones(shape,dtype= np.int64))

    print(np.ones(shape,dtype= np.float64))

    print(np.ones(shape))

@hipy.compiled_function
def fn_ones():
    fn_ones_(3)
    fn_ones_((3,))
    fn_ones_((3,3))

def test_ones():
    check_prints(fn_ones, """
[1 1 1]
[1. 1. 1.]
[1. 1. 1.]
[1 1 1]
[1. 1. 1.]
[1. 1. 1.]
[[1 1 1]
 [1 1 1]
 [1 1 1]]
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
""")

@hipy.compiled_function
def fn_np_array():
    print(np.array([1,2,3]))
    print(np.array([[1,2,3],[4,5,6]]))

def test_np_array():
    check_prints(fn_np_array, """
[1 2 3]
[[1 2 3]
 [4 5 6]]
""")


@hipy.compiled_function
def fn_np_array_concrete():
    arr = np.array([1, 2, 3])
    print(intrinsics.isa(arr, _concrete_ndarray))
    print(arr[0])
    print(intrinsics.isa(arr, _concrete_ndarray))
    print(arr)
    arr = np.array(not_constant([1, 2, 3]))
    print(intrinsics.isa(arr, _concrete_ndarray))

def test_np_array_concrete():
    check_prints(fn_np_array_concrete, """
True
1
True
[1 2 3]
False
""")