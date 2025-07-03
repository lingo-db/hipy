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
