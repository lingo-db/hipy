import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np


@hipy.compiled_function
def fn_get_set_get_1d():
    arr = np.ones(3, dtype=np.int64)
    print(arr[0])
    print(arr[1])
    print(arr[2])
    arr[0] = 2
    arr[1] = 3
    arr[2] = 4
    print(arr[0])
    print(arr[1])
    print(arr[2])

def test_get_set_get_1d():
    check_prints(fn_get_set_get_1d, """
1
1
1
2
3
4
""")

@hipy.compiled_function
def fn_get_set_get_2d():
    arr= np.ones((3,3), dtype=np.int64)
    print(arr[0,0])
    print(arr[1,2])
    print(arr[2,1])
    arr[0,0] = 2
    arr[2,1] = 3
    arr[1,2] = 4
    print(arr[0,0])
    print(arr[1,2])
    print(arr[2,1])

def test_get_set_get_2d():
    check_prints(fn_get_set_get_2d, """
1
1
1
2
4
3
""")


@hipy.compiled_function
def fn_get_set_view_1d():
    arr = np.ones(5, dtype=np.int64)
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    arr[3] = 4
    arr[4] = 5
    view = arr[1:]
    print(view)
    view[0] = 2
    view[1] = 3
    print(view,arr)
    view[1:-1] = 42
    print(view,arr)

def test_get_set_view_1d():
    check_prints(fn_get_set_view_1d, """
[2 3 4 5]
[2 3 4 5] [1 2 3 4 5]
[ 2 42 42  5] [ 1  2 42 42  5]
""")



@hipy.compiled_function
def fn_get_set_view_2d():
    arr = np.ones((5,5), dtype=np.int64)
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    arr[3] = 4
    arr[4] = 5
    view = arr[1:]
    print(view)
    view[0] = 2
    view[1] = 3
    print(view,arr)
    view[1:-1] = 42
    print(view,arr)

def test_get_set_view_2d():
    check_prints(fn_get_set_view_2d, """
[[2 2 2 2 2]
 [3 3 3 3 3]
 [4 4 4 4 4]
 [5 5 5 5 5]]
[[2 2 2 2 2]
 [3 3 3 3 3]
 [4 4 4 4 4]
 [5 5 5 5 5]] [[1 1 1 1 1]
 [2 2 2 2 2]
 [3 3 3 3 3]
 [4 4 4 4 4]
 [5 5 5 5 5]]
[[ 2  2  2  2  2]
 [42 42 42 42 42]
 [42 42 42 42 42]
 [ 5  5  5  5  5]] [[ 1  1  1  1  1]
 [ 2  2  2  2  2]
 [42 42 42 42 42]
 [42 42 42 42 42]
 [ 5  5  5  5  5]]
    """)

@hipy.compiled_function
def fn_copy_partly():
    arr = np.ones(5, dtype=np.int64)
    arr2 = np.zeros(3, dtype=np.int64)
    arr[1:3] = arr2
    print(arr)

def test_copy_partly():
    check_prints(fn_copy_partly, """
[1 0 0 1 1]
""")