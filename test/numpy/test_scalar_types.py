import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np

@hipy.compiled_function
def fn_construct():
    print(np.int64(42))
    print(np.int64(42.0))
    print(np.float64(42.0))
    print(np.float64(42))

def test_construct():
    check_prints(fn_construct, """
42
42
42.0
42.0
""")

@hipy.compiled_function
def fn_numeric_ops():
    print(np.int64(1)+np.int64(1))

def test_numeric_ops():
    check_prints(fn_numeric_ops, """
2
""")

@hipy.compiled_function
def fn_cast_to_python():
    print(int(np.int64(42)))
    print(float(np.float64(42.0)))
    print(int(np.float64(42.0)))
    print(float(np.int64(42)))

def test_cast_to_python():
    check_prints(fn_cast_to_python, """
42
42.0
42
42.0
""")


@hipy.compiled_function
def fn_int64_compare():
    # np.int64 comparison routes through _cmp_op (scalar.int.compare.*).
    a = np.int64(not_constant(5))
    b = np.int64(not_constant(3))
    print(a == a)
    print(a != b)
    print(a < b)
    print(a > b)
    print(a <= a)
    print(a >= b)


def test_int64_compare():
    check_prints(fn_int64_compare, """
True
True
False
True
True
True
""")


@hipy.compiled_function
def fn_int64_arith():
    # np.int64 +/-/* exercised via _int_op (scalar.int.*).
    a = np.int64(not_constant(10))
    b = np.int64(not_constant(3))
    print(a + b)
    print(a - b)
    print(a * b)
    print(a % b)


def test_int64_arith():
    check_prints(fn_int64_arith, """
13
7
30
1
""")


@hipy.compiled_function
def fn_float64_arith():
    # np.float64 +/-/*/pow via _float_op (scalar.float.*).
    a = np.float64(not_constant(2.0))
    b = np.float64(not_constant(3.0))
    print(a + b)
    print(a - b)
    print(a * b)
    print(a ** b)


def test_float64_arith():
    check_prints(fn_float64_arith, """
5.0
-1.0
6.0
8.0
""")


@hipy.compiled_function
def fn_float64_round():
    # np.float64.__round__ returns a scalar via scalar.float.round.
    x = np.float64(not_constant(3.14159))
    print(round(x))
    print(round(x, 2))


def test_float64_round():
    check_prints(fn_float64_round, """
3
3.14
""")