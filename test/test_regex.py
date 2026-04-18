
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