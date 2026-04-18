import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_same():
    print(not_constant(1) + 1)
    print(not_constant(1.0) + 1.0)
    print(not_constant(1) - 2)
    print(not_constant(1.0) - 2.5)
    print(not_constant(3) * 3)
    print(not_constant(1.1) * 1.1)
    print(not_constant(15) / 5)
    print(not_constant(1.0) / 3.0)
    print(not_constant(1) == 1)
    print(not_constant(1) == 2)
    print(not_constant(1) != 1)
    print(not_constant(1) != 2)
    print(not_constant(1) < 1)
    print(not_constant(1) < 2)
    print(not_constant(1) <= 0)
    print(not_constant(1) <= 1)
    print(not_constant(1) <= 2)
    print(not_constant(1) > 0)
    print(not_constant(1) > 1)
    print(not_constant(1) >= 0)
    print(not_constant(1) >= 1)
    print(not_constant(1) >= 2)


def test_numeric_same():
    check_prints(fn_same, """
2
2.0
-1
-1.5
9
1.21
3.0
0.333333
True
False
False
True
False
True
False
True
True
True
False
True
True
False
""")


@hipy.compiled_function
def fn_aug_assign():
    x = not_constant(1)
    x += 1
    print(x)
    x = not_constant(1.0)
    x += 1.0
    print(x)
    x = not_constant(1)
    x -= 2
    print(x)
    x = not_constant(1.0)
    x -= 2.5
    print(x)
    x = not_constant(3)
    x *= 3
    print(x)
    x = not_constant(1.1)
    x *= 1.1
    print(x)
    x = not_constant(15)
    x /= 5
    print(x)
    x = not_constant(1.0)
    x /= 3.0
    print(x)
    x = not_constant(5)
    x %= 3
    print(x)


def test_aug_assign():
    check_prints(fn_aug_assign, """
2
2.0
-1
-1.5
9
1.21
3.0
0.333333
2
""")


@hipy.compiled_function
def fn_int_float():
    i = not_constant(1)
    f = not_constant(1.0)
    print(i + f)
    print(i - f)
    print(i * f)
    print(i / f)
    print(i == f)
    print(i != f)
    print(i < f)
    print(i <= f)
    print(i > f)
    print(i >= f)


def test_int_float():
    check_prints(fn_int_float, """2.0
0.0
1.0
1.0
True
False
False
True
False
True
""")


@hipy.compiled_function
def fn_int_unary_neg():
    # int.__neg__ is defined as `0 - self` but was not exercised for abstract values.
    x = not_constant(5)
    print(-x)
    y = not_constant(-3)
    print(-y)


def test_int_unary_neg():
    check_prints(fn_int_unary_neg, """
-5
3
""")


import pytest


@hipy.compiled_function
def fn_int_invert():
    # int.__invert__ is defined as `self._int_op("xor", -1)` but the C++ backend
    # has no handler for `scalar.int.xor` — exposes a missing-handler bug.
    print(~not_constant(5))


def test_int_invert():
    check_prints(fn_int_invert, """
-6
""")


@hipy.compiled_function
def fn_int_bitwise_abstract():
    # int._int_op("and", ...), "lshift" for abstract operands.
    a = not_constant(0b1100)
    b = not_constant(0b1010)
    print(a & b)
    print(a << not_constant(2))


def test_int_bitwise_abstract():
    check_prints(fn_int_bitwise_abstract, """
8
48
""")


@hipy.compiled_function
def fn_int_cast_from_bool():
    # int(<bool>) should dispatch through int._cast_to_int → bool.__int__ (1/0).
    print(int(not_constant(True)))
    print(int(not_constant(False)))


def test_int_cast_from_bool():
    check_prints(fn_int_cast_from_bool, """
1
0
""")


@hipy.compiled_function
def fn_int_floordiv():
    # Integer floor division was not exercised (only truediv and mod were).
    print(not_constant(7) // 2)
    print(not_constant(-7) // 2)
    print(not_constant(10) // not_constant(3))


@pytest.mark.xfail(reason="scalar.int.div uses C-style truncation; Python floor division requires -7//2 == -4 but currently returns -3")
def test_int_floordiv():
    check_prints(fn_int_floordiv, """
3
-4
3
""")


@hipy.compiled_function
def fn_int_cast_from_string():
    # int(<str>) routes to the scalar.int.from_string builtin.
    s = not_constant("42")
    print(int(s))
    print(int("-17"))


def test_int_cast_from_string():
    check_prints(fn_int_cast_from_string, """
42
-17
""")


@hipy.compiled_function
def fn_float_unary_neg():
    # float.__neg__ uses scalar.float.neg — previously untested for abstract.
    x = not_constant(1.5)
    print(-x)
    y = not_constant(-2.25)
    print(-y)


def test_float_unary_neg():
    check_prints(fn_float_unary_neg, """
-1.5
2.25
""")


@hipy.compiled_function
def fn_float_pow():
    # float.__pow__ via scalar.float.pow (runtime operands).
    a = not_constant(2.0)
    b = not_constant(3.0)
    print(a ** b)
    print(not_constant(4.0) ** 0.5)


def test_float_pow():
    check_prints(fn_float_pow, """
8.0
2.0
""")


@hipy.compiled_function
def fn_float_from_string():
    # float(<str>) is implemented but untested; exercises scalar.float.from_string.
    s = not_constant("3.25")
    print(float(s))
    print(float("-0.5"))


@pytest.mark.xfail(reason="scalar.float.from_string not implemented in C++ backend — float(<str>) broken")
def test_float_from_string():
    check_prints(fn_float_from_string, """
3.25
-0.5
""")


@hipy.compiled_function
def fn_float_format():
    # Float __format__ with ".Nf" spec uses translate_python_spec_to_cpp.
    x = not_constant(3.14159)
    print(f"{x:.2f}")
    print(f"{x:.4f}")


@pytest.mark.xfail(reason="scalar.string.format_single not implemented in C++ backend (only in MLIR) — float.__format__ with format spec broken")
def test_float_format():
    check_prints(fn_float_format, """
3.14
3.1416
""")
