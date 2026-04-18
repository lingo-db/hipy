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


@hipy.compiled_function
def fn_array_sub_div_pow():
    # ndarray __sub__ / __truediv__ / __pow__ — paths not covered by existing scalar/binary ops tests.
    arr = np.ones(3, dtype=np.float64) * 4.0
    print(arr - 1.0)
    print(arr / 2.0)
    print(arr ** 0.5)


def test_array_sub_div_pow():
    check_prints(fn_array_sub_div_pow, """
[3. 3. 3.]
[2. 2. 2.]
[2. 2. 2.]
""")


@hipy.compiled_function
def fn_array_compare():
    # ndarray comparison ops return ndarray<bool> via element-wise lambda.
    arr = np.array([1, 2, 3])
    print(arr == 2)
    print(arr != 2)
    print(arr < 2)


def test_array_compare():
    check_prints(fn_array_compare, """
[False  True False]
[ True False  True]
[ True False False]
""")


@hipy.compiled_function
def fn_array_topython_roundtrip():
    # ndarray.__topython__ via intrinsics.to_python — verifies arrow→CPython array conversion.
    arr = np.array(not_constant([10, 20, 30]))
    p = intrinsics.to_python(arr)
    print(p)


def test_array_topython_roundtrip():
    check_prints(fn_array_topython_roundtrip, """
[10 20 30]
""")


@hipy.compiled_function
def fn_zeros_basic():
    # np.zeros (int64 + float64 overloads).
    print(np.zeros(3, dtype=np.int64))
    print(np.zeros(3, dtype=np.float64))


def test_zeros_basic():
    check_prints(fn_zeros_basic, """
[0 0 0]
[0. 0. 0.]
""")


@hipy.compiled_function
def fn_isnan_float():
    # numpy.isnan on a non-nan float64 — uses scalar.float.isnan C++ builtin.
    print(np.isnan(np.float64(1.0)))
    print(np.isnan(np.float64(0.0)))


def test_isnan_float():
    check_prints(fn_isnan_float, """
False
False
""")