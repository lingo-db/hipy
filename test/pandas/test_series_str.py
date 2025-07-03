import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.pandas
import pandas as pd

@hipy.compiled_function
def fn_slice():
    df = pd.DataFrame.from_dict({"str": ['astr', 'bstr', 'cstr', 'dstr ']})
    print(df['str'].str.slice(0, 1))


def test_slice():
    check_prints(fn_slice, """
0    a
1    b
2    c
3    d
dtype: object""")

@hipy.compiled_function
def fn_lower():
    df = pd.DataFrame.from_dict({"str": ['A', 'b', 'c', 'xD ']})
    print(df['str'].str.lower())


def test_lower():
    check_prints(fn_lower, """
0      a
1      b
2      c
3    xd 
dtype: object""")
@hipy.compiled_function
def fn_contains():
    df = pd.DataFrame.from_dict({"str": ['hello', 'world', 'hallo', 'welt ']})
    print(df['str'].str.contains('llo'))


def test_contains():
    check_prints(fn_contains, """
0     True
1    False
2     True
3    False
dtype: bool
""")

@hipy.compiled_function
def fn_len():
    df = pd.DataFrame.from_dict({"str": ['hello', 'foo', 'bar', 'welt']})
    print(df['str'].str.len())


def test_len():
    check_prints(fn_len, """
0    5
1    3
2    3
3    4
dtype: int64
""")


