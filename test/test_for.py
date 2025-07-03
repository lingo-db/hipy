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