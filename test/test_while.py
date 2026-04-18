import pytest

import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.compiler


@hipy.compiled_function
def fn_while():
    x = 1
    while x < 10:
        x = x + 1
        print(x)
def test_for_tuple_target():
    check_prints(fn_while, """2
3
4
5
6
7
8
9
10""")


@hipy.compiled_function
def fn_while_break():
    # while with break exits early — abstract condition forces runtime loop.
    x = not_constant(0)
    while x < 100:
        if x == 5:
            break
        x += 1
    print(x)


def test_while_break():
    check_prints(fn_while_break, """
5
""")


@hipy.compiled_function
def fn_while_continue():
    # while with continue skips the printed counter increment; abstract bound.
    x = not_constant(0)
    printed = 0
    while x < 6:
        x += 1
        if x % 2 == 0:
            continue
        printed += 1
    print(printed)


def test_while_continue():
    check_prints(fn_while_continue, """
3
""")


@hipy.compiled_function
def fn_while_abstract_cond():
    # Condition references a runtime-abstract value — forces abstract loop.
    n = not_constant(4)
    x = 0
    while x < n:
        x += 1
    print(x)


def test_while_abstract_cond():
    check_prints(fn_while_abstract_cond, """
4
""")
