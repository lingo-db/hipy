import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.math
import math


@hipy.compiled_function
def fn_sqrt():
    # math.sqrt → scalar.float.sqrt; use perfect squares to avoid precision mismatch.
    print(math.sqrt(4.0))
    print(math.sqrt(not_constant(9.0)))
    print(math.sqrt(not_constant(16.0)))


def test_sqrt():
    check_prints(fn_sqrt, """
2.0
3.0
4.0
""")


@hipy.compiled_function
def fn_exp_log():
    # math.exp / math.log → scalar.float.exp / scalar.float.log. exp(0) and log(1) return exact 1.0/0.0.
    print(math.exp(not_constant(0.0)))
    print(math.log(not_constant(1.0)))


def test_exp_log():
    check_prints(fn_exp_log, """
1.0
0.0
""")


@hipy.compiled_function
def fn_trig():
    # sin(0)=0, cos(0)=1 — exact values avoid 6-digit print-precision comparisons.
    print(math.sin(not_constant(0.0)))
    print(math.cos(not_constant(0.0)))


def test_trig():
    check_prints(fn_trig, """
0.0
1.0
""")


import pytest


@hipy.compiled_function
def fn_acos():
    print(math.acos(not_constant(1.0)))


def test_acos():
    check_prints(fn_acos, """
0.0
""")


@hipy.compiled_function
def fn_atan2():
    print(math.atan2(not_constant(0.0), not_constant(1.0)))


def test_atan2():
    check_prints(fn_atan2, """
0.0
""")


@hipy.compiled_function
def fn_ceil():
    # math.ceil → scalar.float.ceil, returns int.
    print(math.ceil(not_constant(1.5)))
    print(math.ceil(not_constant(2.0)))
    print(math.ceil(not_constant(-0.1)))


@pytest.mark.xfail(reason="scalar.float.ceil not implemented in C++ backend")
def test_ceil():
    check_prints(fn_ceil, """
2
2
0
""")


@hipy.compiled_function
def fn_fact():
    # fact from math shim: 1 * 2 * ... * (l) (note: its loop starts at 1 up to l-1 and multiplies i+1).
    print(math.fact(1))
    print(math.fact(5))


def test_fact():
    check_prints(fn_fact, """
1
120
""")


@hipy.compiled_function
def fn_pi_constant():
    # math.pi is a module-level float constant.
    print(math.pi > 3.0)
    print(math.pi < 4.0)


def test_pi_constant():
    check_prints(fn_pi_constant, """
True
True
""")
