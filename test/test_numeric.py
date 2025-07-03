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
