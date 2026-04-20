"""Augmented-assignment operators whose `__i<op>__` dunder is absent
on the receiver type.

Python semantics: `x <op>= y` tries `x.__i<op>__(y)`; if that isn't
defined it falls back to `x = x <op> y`. hipy's rewriter previously
emitted the in-place dunder unconditionally, so any `<op>=` on a type
that didn't define it blew up with AttributeError. `context.perform_augop`
now performs the same fallback as CPython.
"""
import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_ior():
    x = not_constant(5)
    x |= 3
    print(x)


def test_ior():
    check_prints(fn_ior, "7")


@hipy.compiled_function
def fn_ixor():
    x = not_constant(5)
    x ^= 3
    print(x)


def test_ixor():
    check_prints(fn_ixor, "6")


@hipy.compiled_function
def fn_ilshift():
    x = not_constant(5)
    x <<= 2
    print(x)


def test_ilshift():
    check_prints(fn_ilshift, "20")


@hipy.compiled_function
def fn_irshift():
    x = not_constant(20)
    x >>= 2
    print(x)


def test_irshift():
    check_prints(fn_irshift, "5")


@hipy.compiled_function
def fn_ifloordiv():
    x = not_constant(10)
    x //= 3
    print(x)


def test_ifloordiv():
    check_prints(fn_ifloordiv, "3")
