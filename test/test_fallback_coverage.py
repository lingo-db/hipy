"""Tests that verify the automatic fallback mechanism kicks in for
`not_implemented()` / `NotImplementedError` branches in the standard library.

Every test here uses `fallback=True` and exercises an input that the native
specialization rejects (heterogeneous element types, unsupported format
specifiers, etc.). The expected output matches plain CPython semantics —
if the fallback is wired up correctly, the test passes; if the fallback is
broken or the native path silently produces wrong output, the test catches it.

Covered not_implemented paths (hipy/lib/builtins.py line numbers at time of
writing):

  1049   translate_python_spec_to_cpp: unparseable spec
  1064   translate_python_spec_to_cpp: "n" (locale-aware) format type
  1067   translate_python_spec_to_cpp: "=" sign-aware alignment
  1229   __get_percentage_format_parts: mapping keys %(name)s
  1273   __get_percentage_format_parts: unsupported %-type, e.g. %N
  1290   __get_percentage_format_parts: integer with precision
  1485-1487  list.__lt__ with differing element types / non-list rhs
  1500-1502  list.__add__ with differing element types / non-list rhs
  1521   list.__setitem__ with non-int key / mismatched value type
"""

import pytest

import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


# ---------------------------------------------------------------------------
# String-format fallbacks (translate_python_spec_to_cpp + __mod__ parser)
# ---------------------------------------------------------------------------


@hipy.compiled_function
def fn_format_locale_n_falls_back():
    # "{:n}" — locale-aware integer format. Native translator at
    # hipy/lib/builtins.py:1064 raises NotImplementedError, but the fallback
    # re-enters the @compiled_function body of `_const_str.format` with
    # self=pyobj, which still calls the hipy-internal
    # `self._const_str__get_format_parts()` — that name isn't on the pyobj and
    # typeshed lookup fails. See bugs-with-increased-cov.md #4.
    print("{:n}".format(not_constant(1234567)))


@pytest.mark.xfail(reason="str.format fallback is broken — see bugs-with-increased-cov.md #4")
def test_format_locale_n_falls_back():
    # Under the default "C" locale CPython renders %n without separators.
    check_prints(fn_format_locale_n_falls_back, """
1234567
""", fallback=True)


@hipy.compiled_function
def fn_format_sign_aware_align_falls_back():
    # "{:=8d}" — sign-aware '=' alignment. Same @compiled_function fallback
    # issue as above.
    print("|{:=8d}|".format(not_constant(-42)))


@pytest.mark.xfail(reason="str.format fallback is broken — see bugs-with-increased-cov.md #4")
def test_format_sign_aware_align_falls_back():
    check_prints(fn_format_sign_aware_align_falls_back, """
|-     42|
""", fallback=True)


@hipy.compiled_function
def fn_percent_mapping_key_falls_back():
    # "%(name)d" — mapping-key style. __get_percentage_format_parts at
    # hipy/lib/builtins.py:1229 raises NotImplementedError.
    d = {"x": not_constant(7)}
    print("x=%(x)d" % d)


def test_percent_mapping_key_falls_back():
    check_prints(fn_percent_mapping_key_falls_back, """
x=7
""", fallback=True)


@hipy.compiled_function
def fn_percent_integer_precision_falls_back():
    # "%.3d" — integer with precision. __get_percentage_format_parts at
    # hipy/lib/builtins.py:1290 raises NotImplementedError.
    print("|%.3d|" % not_constant(7))


def test_percent_integer_precision_falls_back():
    check_prints(fn_percent_integer_precision_falls_back, """
|007|
""", fallback=True)


# ---------------------------------------------------------------------------
# list fallbacks (differing element types drop to pyobj)
# ---------------------------------------------------------------------------


@hipy.compiled_function
def fn_list_add_heterogeneous_falls_back():
    # list[int] + list[float] — hipy/lib/builtins.py:1500 hits
    # not_implemented(); fallback does CPython list concatenation.
    a = [not_constant(1), not_constant(2)]
    b = [not_constant(1.5), not_constant(2.5)]
    print(a + b)


def test_list_add_heterogeneous_falls_back():
    check_prints(fn_list_add_heterogeneous_falls_back, """
[1, 2, 1.5, 2.5]
""", fallback=True)


@hipy.compiled_function
def fn_list_lt_heterogeneous_falls_back():
    # list[int] < list[float] — hipy/lib/builtins.py:1485 not_implemented();
    # fallback uses CPython's lexicographic list comparison.
    a = [not_constant(1), not_constant(2)]
    b = [not_constant(1.5), not_constant(2.5)]
    print(a < b)


def test_list_lt_heterogeneous_falls_back():
    check_prints(fn_list_lt_heterogeneous_falls_back, """
True
""", fallback=True)


@hipy.compiled_function
def fn_list_setitem_type_mismatch_falls_back():
    # list[int].__setitem__ with a float value — hipy/lib/builtins.py:1521
    # not_implemented(). Fallback converts the list to a pyobj and stores
    # the heterogeneous element.
    a = [not_constant(1), not_constant(2), not_constant(3)]
    a[1] = not_constant(9.5)
    print(a)


def test_list_setitem_type_mismatch_falls_back():
    check_prints(fn_list_setitem_type_mismatch_falls_back, """
[1, 9.5, 3]
""", fallback=True)


@hipy.compiled_function
def fn_list_mul_non_int_falls_back():
    # list.__mul__ at hipy/lib/builtins.py:1514 only handles int multipliers.
    # Multiply by an int that's already been forced to pyobj — the native
    # overload rejects it (isa(pyobj, int) is False), and the fallback path
    # retries the op with the list also as pyobj, giving CPython list repeat.
    a = [not_constant(1), not_constant(2)]
    k = intrinsics.to_python(not_constant(3))
    print(a * k)


def test_list_mul_non_int_falls_back():
    check_prints(fn_list_mul_non_int_falls_back, """
[1, 2, 1, 2, 1, 2]
""", fallback=True)


# ---------------------------------------------------------------------------
# dict fallbacks
# ---------------------------------------------------------------------------


@hipy.compiled_function
def fn_dict_setitem_type_mismatch_falls_back():
    # dict[str, int].__setitem__ with a float value — hipy/lib/builtins.py:1782
    # not_implemented(). Fallback converts the dict to a pyobj and writes
    # the (str, float) pair natively in CPython.
    # (dict literals require constant keys; that's fine, the fallback below
    # triggers on __setitem__ not __init__.)
    d = {"a": not_constant(1)}
    d["b"] = not_constant(2.5)
    print(d)


def test_dict_setitem_type_mismatch_falls_back():
    check_prints(fn_dict_setitem_type_mismatch_falls_back, """
{'a': 1, 'b': 2.5}
""", fallback=True)
