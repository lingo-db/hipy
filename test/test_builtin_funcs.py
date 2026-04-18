import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant

@hipy.compiled_function
def fn_min():
    print(min([1,2,3]))
    print(min(1,2,3))
    print(min(not_constant([1,2,3])))
    print(min(not_constant(1),not_constant(2),not_constant(3)))


def test_min():
    check_prints(fn_min, """
1
1
1
1""")


@hipy.compiled_function
def fn_max():
    # max over list and varargs, constant and abstract.
    print(max([1, 2, 3]))
    print(max(1, 2, 3))
    print(max(not_constant([1, 2, 3])))
    print(max(not_constant(1), not_constant(2), not_constant(3)))


def test_max():
    check_prints(fn_max, """
3
3
3
3
""")


@hipy.compiled_function
def fn_sum():
    # sum over int list seeds with r=0; float list seeds with r=0.0.
    print(sum([1, 2, 3]))
    print(sum(not_constant([1, 2, 3])))
    print(sum([1.5, 2.5, 3.0]))
    print(sum(not_constant([1.5, 2.5, 3.0])))


def test_sum():
    check_prints(fn_sum, """
6
6
7.0
7.0
""")


@hipy.compiled_function
def fn_abs():
    # abs over int and float, positive and negative, abstract.
    print(abs(not_constant(5)))
    print(abs(not_constant(-5)))
    print(abs(not_constant(0)))
    print(abs(not_constant(1.5)))
    print(abs(not_constant(-2.25)))


def test_abs():
    check_prints(fn_abs, """
5
5
0
1.5
2.25
""")


@hipy.compiled_function
def fn_sorted():
    # sorted materializes a list(input) then .sort()s it.
    print(sorted(not_constant([3, 1, 2])))
    print(sorted([5, 2, 4, 1, 3]))


def test_sorted():
    check_prints(fn_sorted, """
[1, 2, 3]
[1, 2, 3, 4, 5]
""")


@hipy.compiled_function
def fn_round_float():
    # round on float routes to scalar.float.round with/without ndigits.
    # Note: HiPy uses away-from-zero rounding (C std round()), while CPython
    # uses banker's (round half-to-even). Avoid .5 cases to sidestep that diff.
    print(round(not_constant(3.6)))
    print(round(not_constant(3.4)))
    print(round(not_constant(3.14159), 2))
    print(round(not_constant(-2.2)))


def test_round_float():
    check_prints(fn_round_float, """
4
3
3.14
-2
""")


@hipy.compiled_function
def fn_round_int():
    # round(int) with no ndigits short-circuits and returns the int unchanged.
    print(round(not_constant(5)))
    print(round(not_constant(-3)))


def test_round_int():
    check_prints(fn_round_int, """
5
-3
""")


@hipy.compiled_function
def fn_format_default():
    # format() with empty spec calls __format__("") which for str is identity, for int uses str().
    print(format(not_constant("hi")))
    print(format(not_constant(42)))


def test_format_default():
    check_prints(fn_format_default, """
hi
42
""")


@hipy.compiled_function
def fn_range_bounds():
    # range: 1-arg, 2-arg, 3-arg and negative step / empty iteration.
    for i in range(3):
        print(i)
    for i in range(2, 5):
        print(i)
    for i in range(0, 10, 3):
        print(i)
    count = 0
    for i in range(5, 0, -1):
        count += 1
    print(count)
    # Empty range
    count2 = 0
    for i in range(5, 5):
        count2 += 1
    print(count2)


def test_range_bounds():
    check_prints(fn_range_bounds, """
0
1
2
2
3
4
0
3
6
9
5
0
""")


@hipy.compiled_function
def fn_enumerate_basic():
    # enumerate yields (index, value) tuples.
    for i, v in enumerate(not_constant(["a", "b", "c"])):
        print(i, v)


def test_enumerate_basic():
    check_prints(fn_enumerate_basic, """
0 a
1 b
2 c
""")


@hipy.compiled_function
def fn_len_variants():
    # len dispatches through __len__ on multiple types.
    print(len(not_constant("hello")))
    print(len(not_constant([1, 2, 3, 4])))
    print(len(not_constant((1, 2))))
    print(len(not_constant({"a": 1, "b": 2, "c": 3})))


def test_len_variants():
    check_prints(fn_len_variants, """
5
4
2
3
""")


@hipy.compiled_function
def fn_ord_abstract():
    # ord(abstract str) dispatches to scalar.string.ord;
    # ord(const str) uses _const_ord short-circuit.
    print(ord("A"))
    print(ord(not_constant("A")))
    print(ord(not_constant("\n")))


def test_ord_abstract():
    check_prints(fn_ord_abstract, """
65
65
10
""")


@hipy.compiled_function
def fn_slice_object():
    # slice() constructor + use in subscription; step applied via range + join.
    s = not_constant("abcdef")
    sl = slice(1, 4)
    print(s[sl])


def test_slice_object():
    check_prints(fn_slice_object, """
bcd
""")