import pytest
import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
import hipy.compiler
from hipy.test_utils import not_constant

"""
Test1: a (complex) example for correct function calls, where all arguments have type hints
"""


@hipy.compiled_function
def fn1(x):
    print(x)
    return x * 2


@hipy.compiled_function
def fn2(x, y):
    return x - y


@hipy.compiled_function
def fn3(x, y):
    return fn1(fn2(x, y))


@hipy.compiled_function
def fn_call():
    print(fn3(1, 2))


def test_call():
    check_prints(fn_call, """-1
-2""")


"""
Test2: calling a function without type hints with different types (should work)
"""


@hipy.compiled_function
def add(x, y):
    return x + y


@hipy.compiled_function
def fn_call_no_types():
    a = not_constant(0)
    b = not_constant(2)
    print(add(a, b))
    c = not_constant(0.0)
    d = not_constant(2.0)
    print(add(c, d))


def test_call_no_types():
    check_prints(fn_call_no_types, """
2
2.0
""")


"""
Test5: call a dbpy member function 
"""


@hipy.compiled_function
def fn_call_lib_method():
    print("  test  ".strip())


def test_call_lib_method():
    check_prints(fn_call_lib_method, """
test
""")


"""
Test5: calling a function with optional parameters
"""


@hipy.compiled_function
def fn_optional_args(x, y=0, z=0):
    print("---------------")
    print(x)
    print(y)
    print(z)


@hipy.compiled_function
def fn_call_optional_args():
    fn_optional_args(1, 2, 3)
    fn_optional_args(1, 2)
    fn_optional_args(1)
    fn_optional_args(1, z=3)


def test_call_optional_args():
    check_prints(fn_call_optional_args, """
---------------
1
2
3
---------------
1
2
0
---------------
1
0
0
---------------
1
0
3
""")


"""
Test6: calling a function with kw arguments
"""


@hipy.compiled_function
def fn_kw_args(x, y=0, z=0):
    print("---------------")
    print(x)
    print(y)
    print(z)


@hipy.compiled_function
def fn_call_kw_args():
    fn_kw_args(z=2, x=0, y=1)


def test_call_kw_args():
    check_prints(fn_call_kw_args, """
---------------
0
1
2
""")


@hipy.compiled_function
def fn_call_varargs_(*args):
    print(args)
    print(*args)


@hipy.compiled_function
def fn_call_varargs():
    fn_call_varargs_()
    fn_call_varargs_(1)
    fn_call_varargs_(1, 2)
    fn_call_varargs_(1, 2, 3)


def test_call_varargs():
    check_prints(fn_call_varargs, """
()

(1,)
1
(1, 2)
1 2
(1, 2, 3)
1 2 3
""")


@hipy.compiled_function
def rec(x, counter=0):
    print("rec", x, counter)
    if x:
        return rec(False, counter + 1)
    else:
        return x


@hipy.compiled_function
def fn_rec():
    z_true = rec(True)
    print(z_true, intrinsics.isa(z_true, bool))
    print("----------------")
    z_false = rec(False)
    print(z_false, intrinsics.isa(z_false, bool))

def test_rec():
    check_prints(fn_rec, """
rec True 0
rec False 1
False True
----------------
rec False 0
False True
""")
