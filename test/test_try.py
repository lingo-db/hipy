import pytest

import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant



def foo() -> int:
    raise Exception("foo")


def bar() -> int:
    return 1


def divide(a, b) -> float:
    return a / b


@hipy.compiled_function
def fn_try_or_default():
    print(intrinsics.try_or_default(lambda: foo(), 0))  # foo raises exception
    print(intrinsics.try_or_default(lambda: bar(), 42))
    a = 1.0
    print(intrinsics.try_or_default(lambda: divide(a, 0.0), 0.0))  # division by zero
    print(intrinsics.try_or_default(lambda: divide(a, 2.0), 0.0))


def test_try_or_default():
    check_prints(fn_try_or_default, """0
1
0.0
0.5
""")


@hipy.compiled_function
def fn_try_print():
    try:
        print(foo())
    except:
        print("caught")
    try:
        print(bar())
    except:
        print("caught")
    try:
        print(divide(1.0, 0.0))
    except:
        print("caught")
    try:
        print(divide(1.0, 2.0))
    except:
        print("caught")


def test_try_print():
    check_prints(fn_try_print, """caught
1
caught
0.5
""")

@hipy.compiled_function
def try_int(val):
    try:
        return int(val)
    except:
        return 0

@hipy.compiled_function
def fn_try_return():
    print(try_int("42"))
    print(try_int("abc"))

def test_try_return():
    check_prints(fn_try_return, """42
0""")


@hipy.compiled_function
def fn_try_assign():
    try:
        x = try_int("abc")
    except:
        x=0
    print(x)
    try:
        x = try_int("42")
    except:
        x=0
    print(x)

def test_try_assign():
    check_prints(fn_try_assign, """0
42
""")


@hipy.compiled_function
def fn_try_abstract_int():
    # int(<abstract str>) can raise at runtime → except catches it.
    s = not_constant("abc")
    try:
        v = int(s)
    except:
        v = -1
    print(v)
    s2 = not_constant("100")
    try:
        v2 = int(s2)
    except:
        v2 = -1
    print(v2)


def test_try_abstract_int():
    check_prints(fn_try_abstract_int, """
-1
100
""")


@hipy.compiled_function
def fn_try_nested():
    # Nested try — inner catches, outer untriggered.
    try:
        try:
            v = int(not_constant("bad"))
        except:
            v = 1
        print(v)
    except:
        print("outer")

    # Nested try — inner except re-raises by calling a raising function, outer catches.
    try:
        try:
            v = int(not_constant("bad"))
        except:
            v = int(not_constant("also-bad"))
    except:
        print("outer")


def test_try_nested():
    check_prints(fn_try_nested, """
1
outer
""")