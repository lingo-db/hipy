import numpy as np

import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant

import time


@hipy.compiled_function
def fn_call_py():
    print(np.complex128(0))


def test_call_numpy():
    check_prints(fn_call_py, """
0j
""",fallback=True)


def foo(x):
    return f"in python: {x}"


@hipy.compiled_function
def fn_call_py_fn():
    print(foo(42))
    print(foo("hello"))


#     print(foo([1, 2, 3]))
#     print(foo({"a": 1, "b": 2}))
#
def test_call_py_fn():
    check_prints(fn_call_py_fn, """
in python: 42
in python: hello""")


# in python: [1, 2, 3]
# in python: {'a': 1, 'b': 2}
# """)
#
# def bar(x) -> str:
#     return str(x)
#
# @dbpyfn
# def fn_call_py_fn_with_return_type():
#     print(bar(42))
# def test_call_py_fn_with_return_type():
#     check_prints(fn_call_py_fn_with_return_type, """42""")
#
#
def kw_fn(a, b) -> int:
    return a + b


@hipy.compiled_function
def fn_call_py_fn_with_kw_args():
    print(kw_fn(1, b=2))


def test_call_py_fn_with_kw_args():
    check_prints(fn_call_py_fn_with_kw_args, """3""")


from hipy import intrinsics


@hipy.compiled_function
def fn_pyobj_arith():
    # object.__add__/__sub__/__mul__/__truediv__ route to python.operator.*.
    # Force a pyobj path by moving an int through to_python, then arithmetic on it.
    a = intrinsics.to_python(not_constant(10))
    b = intrinsics.to_python(not_constant(3))
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)


def test_pyobj_arith():
    check_prints(fn_pyobj_arith, """
13
7
30
3.3333333333333335
""", fallback=True)


@hipy.compiled_function
def fn_pyobj_compare():
    # object.__eq__/__ne__/__lt__/__gt__/__le__/__ge__ route through python.operator.*.
    a = intrinsics.to_python(not_constant(5))
    b = intrinsics.to_python(not_constant(10))
    print(a == b)
    print(a != b)
    print(a < b)
    print(a > b)
    print(a <= b)
    print(a >= b)


def test_pyobj_compare():
    check_prints(fn_pyobj_compare, """
False
True
True
False
True
False
""", fallback=True)


@hipy.compiled_function
def fn_pyobj_pow_small_int():
    # object.__pow__: the branch for _const_int < 5 unrolls via repeated multiply.
    a = intrinsics.to_python(not_constant(2))
    print(a ** 3)  # 2*2*2 unrolled
    print(a ** 4)  # unrolled


def test_pyobj_pow_small_int():
    check_prints(fn_pyobj_pow_small_int, """
8
16
""", fallback=True)


import pytest


@hipy.compiled_function
def fn_pyobj_pow_fallback():
    # object.__pow__ with a non-const int exponent goes through python.operator.pow.
    a = intrinsics.to_python(not_constant(2))
    e = intrinsics.to_python(not_constant(10))
    print(a ** e)


def test_pyobj_pow_fallback():
    check_prints(fn_pyobj_pow_fallback, """
1024
""", fallback=True)


@hipy.compiled_function
def fn_pyobj_getattr_on_module():
    # object.__hipy_getattr__ via PyGetAttr when walking the numpy module.
    # Exercises attribute access + method call on a pyobj without defining
    # a class in the test module. `arr.shape` differs (HiPy ndarray's shape is
    # a HiPy tuple, printed as [3]), so index into it for a stable print.
    arr = np.array([1, 2, 3])
    print(arr.sum())
    print(arr.shape[0])


def test_pyobj_getattr_on_module():
    check_prints(fn_pyobj_getattr_on_module, """
6
3
""", fallback=True)


@hipy.compiled_function
def fn_pyobj_repr():
    # object.__hipy__repr__ calls __repr__() then scalar.string.from_python.
    o = intrinsics.to_python(not_constant([1, 2, 3]))
    print(repr(o))


def test_pyobj_repr():
    check_prints(fn_pyobj_repr, """
[1, 2, 3]
""", fallback=True)


@hipy.compiled_function
def fn_int_topython_roundtrip():
    # Round-trip int through pyobj via scalar.int.to_python + from_python.
    x = not_constant(42)
    p = intrinsics.to_python(x)
    if intrinsics.isa(p, object):
        print("is pyobj")
    # Inference: no abstract_path (came from scalar.int.to_python without annotate),
    # so the result stays as a pyobj and prints via __repr__.
    print(p)


def test_int_topython_roundtrip():
    check_prints(fn_int_topython_roundtrip, """
is pyobj
42
""", fallback=True)


@hipy.compiled_function
def fn_float_topython_roundtrip():
    x = not_constant(3.14)
    p = intrinsics.to_python(x)
    if intrinsics.isa(p, object):
        print("is pyobj")
    print(p)


def test_float_topython_roundtrip():
    check_prints(fn_float_topython_roundtrip, """
is pyobj
3.14
""", fallback=True)
