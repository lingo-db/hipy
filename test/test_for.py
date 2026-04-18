import pytest

import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.compiler


@hipy.compiled_function
def fn_for_tuple_target():
    for i, j in [(1, 2), (3, 4)]:
        print(i)
        print(j)
def test_for_tuple_target():
    check_prints(fn_for_tuple_target, """1
2
3
4""")


@hipy.compiled_function
def fn_for_iter_vals():
    x=1
    for i in range(11):
        x=x*(i+1)
    print(x)

def test_for_iter_vals():
    check_prints(fn_for_iter_vals, """39916800""")


@hipy.compiled_function
def fn_for_continue():
    for i in range(not_constant(10)):
        if i%2==0:
            continue
        print(i)

def test_for_continue():
    check_prints(fn_for_continue, """1
3
5
7
9""")


@hipy.compiled_function
def fn_for_break():
    for i in range(not_constant(10)):
        if i==5:
            break
        print(i)

def test_for_break():
    check_prints(fn_for_break, """0
1
2
3
4""")


@hipy.compiled_function
def fn_for_nested():
    # Nested abstract for loops — sum of i*j for i,j in [0..3)x[0..3).
    total = 0
    for i in range(not_constant(3)):
        for j in range(not_constant(3)):
            total += i * j
    print(total)


def test_for_nested():
    check_prints(fn_for_nested, """
9
""")


@hipy.compiled_function
def fn_for_string():
    # Iterating a string yields single-char strings (via str._iterator).
    total = 0
    for c in not_constant("abc"):
        total += ord(c)
    print(total)


def test_for_string():
    check_prints(fn_for_string, """
294
""")


@hipy.compiled_function
def fn_for_enumerate_break():
    # enumerate + break exits both index and value iteration.
    idx = -1
    for i, v in enumerate(not_constant([10, 20, 30, 40])):
        idx = i
        if v == 30:
            break
    print(idx)


def test_for_enumerate_break():
    check_prints(fn_for_enumerate_break, """
2
""")