
import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import re
import hipy.lib.re
@hipy.compiled_function
def extractid(val):
  match = re.search("(\d+)_zpid/$",val)
  try:
      return int(match.group(1))
  except:
      return  0

@hipy.compiled_function
def fn_simple_regex_search():
    print(extractid("123_zpid/"))
    print(extractid("456_zpid/"))
    print(extractid("no_match"))


def test_simple_regex_search():
    check_prints(fn_simple_regex_search, """
123
456
0
""")


@hipy.compiled_function
def fn_regex_multiple_groups():
    # Two capturing groups exercise _count_groups + _const_sized_tuple for n=2.
    m = re.search("(\w+)-(\d+)", not_constant("abc-42"))
    if m is not None:
        print(m.group(1))
        print(m.group(2))
        print(m.group(0))


def test_regex_multiple_groups():
    check_prints(fn_regex_multiple_groups, """
abc
42
abc-42
""")


@hipy.compiled_function
def fn_regex_span():
    # Match.span() returns the (start, end) tuple for group 0.
    m = re.search("(\d+)", not_constant("xx123yy"))
    if m is not None:
        print(m.span())
        print(m.group())


def test_regex_span():
    check_prints(fn_regex_span, """
(2, 5)
123
""")


@hipy.compiled_function
def fn_regex_match_str():
    # Match.__str__ (hipy/lib/re.py:31) — `<Match object; span=..., match='...'>`.
    # Exercised by passing a Match to print() directly (not its .group()).
    m = re.search("(\d+)", not_constant("xx123yy"))
    if m is not None:
        print(str(m))


def test_regex_match_str():
    check_prints(fn_regex_match_str, """
<Match object; span=(2, 5), match='123'>
""")


@hipy.compiled_function
def fn_regex_match_topython():
    # Match.__topython__ (hipy/lib/re.py:27) — fallback path that re-runs the
    # search via the real `re` module. Triggered by handing a Match across
    # the hipy→python boundary via intrinsics.to_python.
    m = re.search("(\d+)", not_constant("abc42def"))
    if m is not None:
        py_m = intrinsics.to_python(m)
        print(py_m.group(0))
        print(py_m.span())


def test_regex_match_topython():
    check_prints(fn_regex_match_topython, """
42
(3, 5)
""", fallback=True)