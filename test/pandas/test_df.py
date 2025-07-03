import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.pandas
import pandas as pd

from hipy.value import raw_module

_pd = raw_module(pd)


@hipy.compiled_function
def fn_from_dict():
    df = pd.DataFrame.from_dict({"a": [0, 1, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']})
    print(df)

def test_from_dict():
    check_prints(fn_from_dict, """
   a    b  c
0  0  0.0  a
1  1  1.1  b
2  2  2.2  c
3  3  3.3  d
""")

@hipy.compiled_function
def fn_constructor():
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']})
    print(df)
    print(df.index)
    df_with_index=pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']}, index=pd.Index(['a', 'b', 'c', 'd'], name='index'))
    print(df_with_index)
    print(df_with_index.index)

def test_constructor():
    check_prints(fn_constructor, """
   a    b  c
0  0  0.0  a
1  1  1.1  b
2  2  2.2  c
3  3  3.3  d
RangeIndex(start=0, stop=4, step=1)
       a    b  c
index           
a      0  0.0  a
b      1  1.1  b
c      2  2.2  c
d      3  3.3  d
Index(['a', 'b', 'c', 'd'], dtype='object', name='index')
""")

@hipy.compiled_function
def fn_sort_values():
    df = pd.DataFrame.from_dict({"a": [0, 2, 2, 1], "b": [2.2, 0.0, 1.1, 3.3], "c": ['a', 'c', 'b', 'd']})
    print(df.sort_values(by='a'))
    print(df.sort_values(by='a', ascending=False))
    print(df.sort_values(by=['a', 'b'], ascending=False))
    print(df.sort_values(by=['a', 'b'], ascending=[False, True]))

def test_sort_values():
    check_prints(fn_sort_values, """
   a    b  c
0  0  2.2  a
3  1  3.3  d
1  2  0.0  c
2  2  1.1  b
   a    b  c
1  2  0.0  c
2  2  1.1  b
3  1  3.3  d
0  0  2.2  a
   a    b  c
2  2  1.1  b
1  2  0.0  c
3  1  3.3  d
0  0  2.2  a
   a    b  c
1  2  0.0  c
2  2  1.1  b
3  1  3.3  d
0  0  2.2  a
    """)

@hipy.compiled_function
def fn_filter():
    df = pd.DataFrame.from_dict({"a": [0, 1, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']})
    print(df[df['a'] == 1])


def test_filter():
    check_prints(fn_filter, """
   a    b  c
1  1  1.1  b
""")

@hipy.compiled_function
def fn_slice():
    df = pd.DataFrame.from_dict({"a": [0, 1, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']})
    print(df[1:-1])


def test_slice():
    check_prints(fn_slice, """
   a    b  c
1  1  1.1  b
2  2  2.2  c
""")

@hipy.compiled_function
def fn_select_columns():
    df = pd.DataFrame.from_dict({"a": [0, 1, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']})
    print(df[['a', 'b']])


def test_select_columns():
    check_prints(fn_select_columns, """
   a    b
0  0  0.0
1  1  1.1
2  2  2.2
3  3  3.3
""")

@hipy.compiled_function
def fn_set_columns():
    df = pd.DataFrame.from_dict({"a": [0, 1, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']})
    df['a'] = df['c']
    df['d'] = df['a']
    df['e'] = pd.Series([0, 1, 2, 3])
    print(df)


def test_set_columns():
    check_prints(fn_set_columns, """
   a    b  c  d  e
0  a  0.0  a  a  0
1  b  1.1  b  b  1
2  c  2.2  c  c  2
3  d  3.3  d  d  3
""")

@hipy.compiled_function
def fn_get_column():
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']}, index=pd.Index(['a', 'b', 'c', 'd'], name='index'))
    print(df['a'])

def test_get_column():
    check_prints(fn_get_column, """
index
a    0
b    1
c    2
d    3
Name: a, dtype: int64
""")


@hipy.compiled_function
def fn_iloc_int():
    df = pd.DataFrame.from_dict({"a": [1, 0, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']})
    print(df.iloc[0])
    print(df.iloc[1])
    print(df.iloc[2])

def test_iloc_int():
    check_prints(fn_iloc_int, """
a      1
b    0.0
c      a
Name: 0, dtype: object
a      0
b    1.1
c      b
Name: 1, dtype: object
a      2
b    2.2
c      c
Name: 2, dtype: object
""")

@hipy.compiled_function
def fn_iloc_slice():
    df = pd.DataFrame.from_dict({"a": [1, 0, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']})
    print(df.iloc[0:2])
    print(df.iloc[1:])
    print(df.iloc[:2])

def test_iloc_slice():
    check_prints(fn_iloc_slice, """
   a    b  c
0  1  0.0  a
1  0  1.1  b
   a    b  c
1  0  1.1  b
2  2  2.2  c
3  3  3.3  d
   a    b  c
0  1  0.0  a
1  0  1.1  b
""")


@hipy.compiled_function
def fn_apply():
    _pd.set_option('display.expand_frame_repr', False)
    _pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame({"a": [1, 0, 2, 3], "b": [0.0, 1.1, 2.2, 3.3], "c": ['a', 'b', 'c', 'd']}, index=pd.Index(['a', 'b', 'c', 'd'], name='index'))
    print(df.apply(lambda x: x.apply(lambda y: str(y) + "_blubb")))
    print(df.apply(lambda x: str(x), axis=1))
    print(df.apply(lambda x: str(x['a'])+"_"+str(x['b'])+ "_"+x['c'], axis=1))


def test_apply():
    check_prints(fn_apply,"""
             a          b        c
index                             
a      1_blubb  0.0_blubb  a_blubb
b      0_blubb  1.1_blubb  b_blubb
c      2_blubb  2.2_blubb  c_blubb
d      3_blubb  3.3_blubb  d_blubb
index
a    a      1\\nb    0.0\\nc      a\\nName: a, dtype: object
b    a      0\\nb    1.1\\nc      b\\nName: b, dtype: object
c    a      2\\nb    2.2\\nc      c\\nName: c, dtype: object
d    a      3\\nb    3.3\\nc      d\\nName: d, dtype: object
dtype: object
index
a    1_0.0_a
b    0_1.1_b
c    2_2.2_c
d    3_3.3_d
dtype: object
    """)