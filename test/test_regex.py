
import hipy
from hipy.interpreter import check_prints
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