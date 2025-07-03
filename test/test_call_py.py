import numpy as np

import hipy
from hipy.interpreter import check_prints

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
