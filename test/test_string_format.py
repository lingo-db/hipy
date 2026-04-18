"""Tests for str.format(...) and str % ... .

These exercise `_const_str.translate_python_spec_to_cpp`,
`_const_str.__get_format_parts`, and `_const_str.__get_percentage_format_parts`
in `hipy/lib/builtins.py`. The entire format-string parser was previously
untested — each `check_prints` below walks a specific branch of the parser.
"""

import pytest

import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_format_basic():
    # Empty spec, just default str().
    print("{}".format(not_constant(42)))
    print("{}".format(not_constant("hi")))
    # Multiple fields, mixed types.
    print("a={} b={}".format(not_constant(1), not_constant(2)))
    # Pure literal, no fields.
    print("no fields".format())


def test_format_basic():
    check_prints(fn_format_basic, """
42
hi
a=1 b=2
no fields
""")


@hipy.compiled_function
def fn_format_literal_only():
    # Literal-only path through the parser (no specs).
    print("no fields".format())


def test_format_literal_only():
    check_prints(fn_format_literal_only, """
no fields
""")


@hipy.compiled_function
def fn_format_integer_specs():
    # Width, zero-pad, sign.
    print("|{:5d}|".format(not_constant(7)))
    print("|{:05d}|".format(not_constant(7)))
    print("|{:+d}|".format(not_constant(7)))
    print("|{:+d}|".format(not_constant(-7)))
    print("|{: d}|".format(not_constant(7)))
    # Alternate form for hex/oct/bin.
    print("|{:#x}|".format(not_constant(255)))
    print("|{:#o}|".format(not_constant(8)))
    print("|{:#b}|".format(not_constant(5)))


def test_format_integer_specs():
    check_prints(fn_format_integer_specs, """
|    7|
|00007|
|+7|
|-7|
| 7|
|0xff|
|010|
|0b101|
""")


@hipy.compiled_function
def fn_format_float_specs():
    # Precision and fixed/exponential/general.
    print("|{:.3f}|".format(not_constant(3.14159)))
    print("|{:.0f}|".format(not_constant(3.9)))
    print("|{:10.4f}|".format(not_constant(2.5)))
    print("|{:+.2f}|".format(not_constant(2.5)))


def test_format_float_specs():
    check_prints(fn_format_float_specs, """
|3.142|
|4|
|    2.5000|
|+2.50|
""")


@hipy.compiled_function
def fn_format_align_fill():
    # align with default fill (space).
    print("|{:<8}|".format(not_constant("hi")))
    print("|{:>8}|".format(not_constant("hi")))
    print("|{:^8}|".format(not_constant("hi")))
    # explicit fill + align.
    print("|{:*<8}|".format(not_constant("hi")))
    print("|{:*>8}|".format(not_constant("hi")))
    print("|{:*^8}|".format(not_constant("hi")))


def test_format_align_fill():
    check_prints(fn_format_align_fill, """
|hi      |
|      hi|
|   hi   |
|hi******|
|******hi|
|***hi***|
""")


@hipy.compiled_function
def fn_percent_integer():
    # Basic %d / %i / %u / %o / %x / %X.
    print("%d" % not_constant(42))
    print("%i" % not_constant(42))
    print("%u" % not_constant(42))
    print("%o" % not_constant(8))
    print("%x" % not_constant(255))
    print("%X" % not_constant(255))


def test_percent_integer():
    check_prints(fn_percent_integer, """
42
42
42
10
ff
FF
""")


@hipy.compiled_function
def fn_percent_flags():
    # Width + flags: '-' left, '+' sign, ' ' sign, '0' zero pad, '#' alt.
    print("|%5d|" % not_constant(7))
    print("|%-5d|" % not_constant(7))
    print("|%05d|" % not_constant(7))
    print("|%+d|" % not_constant(7))
    print("|% d|" % not_constant(7))
    print("|%#x|" % not_constant(255))


def test_percent_flags():
    check_prints(fn_percent_flags, """
|    7|
|7    |
|00007|
|+7|
| 7|
|0xff|
""")


@hipy.compiled_function
def fn_percent_float_and_string():
    print("|%f|" % not_constant(1.5))
    print("|%.3f|" % not_constant(3.14159))
    print("|%10.2f|" % not_constant(2.5))
    print("|%-10.2f|" % not_constant(2.5))
    print("|%e|" % not_constant(12345.6789))
    print("|%.3g|" % not_constant(0.00012345))
    print("hello %s" % not_constant("world"))


def test_percent_float_and_string():
    check_prints(fn_percent_float_and_string, """
|1.500000|
|3.142|
|      2.50|
|2.50      |
|1.234568e+04|
|0.000123|
hello world
""")


@hipy.compiled_function
def fn_percent_single_and_trailing():
    # Trailing literal after last spec and single-arg RHS.
    print("val=%d done" % not_constant(3))
    # %% escape inside a single-spec format (no tuple RHS).
    print("100%% = %d" % not_constant(50))


def test_percent_single_and_trailing():
    check_prints(fn_percent_single_and_trailing, """
val=3 done
100% = 50
""")


@hipy.compiled_function
def fn_percent_tuple_rhs():
    # Python unpacks a tuple RHS into positional args for %.
    print("100%% of %d is %d" % (not_constant(50), not_constant(50)))


@pytest.mark.xfail(reason="tuple RHS is not unpacked — see bugs-with-increased-cov.md #2")
def test_percent_tuple_rhs():
    check_prints(fn_percent_tuple_rhs, """
100% of 50 is 50
""")
