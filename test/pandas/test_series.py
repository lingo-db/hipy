import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.pandas
import pandas as pd
import hipy.lib.numpy
import numpy as np

@hipy.compiled_function
def fn_from_list():
    s = pd.Series([1, 2, 3])
    print(s)
    s = pd.Series([1, 2, 3], index=pd.Index(['a', 'b', 'c'], name='index'))
    print(s)


def test_from_list():
    check_prints(fn_from_list, """
0    1
1    2
2    3
dtype: int64
index
a    1
b    2
c    3
dtype: int64
""")
@hipy.compiled_function
def fn_apply():
    s = pd.Series([1, 2, 3])
    print(s.apply(lambda x: x+1))
    print(s.apply(lambda x: float(x)))
    print(s.apply(lambda x: str(x)))

def test_apply():
    check_prints(fn_apply, """
0    2
1    3
2    4
dtype: int64
0    1.0
1    2.0
2    3.0
dtype: float64
0    1
1    2
2    3
dtype: object
""")


@hipy.compiled_function
def fn_bool_ops():
    s1 = pd.Series([True, False, True, False])
    s2 = pd.Series([True, True, False, False])
    print(s1 & s2)
    print(s1 | s2)
    print(~s1)

def test_ops():
    check_prints(fn_bool_ops, """
0     True
1    False
2    False
3    False
dtype: bool
0     True
1     True
2     True
3    False
dtype: bool
0    False
1     True
2    False
3     True
dtype: bool
    """)



@hipy.compiled_function
def fn_comparisons_scalar():
    s = pd.Series([1, 2, 3])
    print(s == 2)
    print(s != 2)
    print(s > 1)
    print(s < 2)
    print(s >= 2)
    print(s <= 2)

def test_comparisons_scalar():
    check_prints(fn_comparisons_scalar, """
0    False
1     True
2    False
dtype: bool
0     True
1    False
2     True
dtype: bool
0    False
1     True
2     True
dtype: bool
0     True
1    False
2    False
dtype: bool
0    False
1     True
2     True
dtype: bool
0     True
1     True
2    False
dtype: bool
    """)


@hipy.compiled_function
def fn_comparisons_series():
    s = pd.Series([1, 2, 3])
    other = pd.Series([2, 2, 2])
    print(s == other)
    print(s != other)
    print(s > other)
    print(s < other)
    print(s >= other)
    print(s <= other)

def test_comparisons_series():
    check_prints(fn_comparisons_series, """
0    False
1     True
2    False
dtype: bool
0     True
1    False
2     True
dtype: bool
0    False
1    False
2     True
dtype: bool
0     True
1    False
2    False
dtype: bool
0    False
1     True
2     True
dtype: bool
0     True
1     True
2    False
dtype: bool
""")

@hipy.compiled_function
def fn_iloc_int():
    s = pd.Series([1, 2, 3])
    print(s.iloc[0])
    print(s.iloc[1])
    print(s.iloc[2])

def test_iloc_int():
    check_prints(fn_iloc_int, """
1
2
3
""")

@hipy.compiled_function
def fn_iloc_slice():
    s = pd.Series([1, 2, 3])
    print(s.iloc[0:2])
    print(s.iloc[1:])
    print(s.iloc[:2])

def test_iloc_slice():
    check_prints(fn_iloc_slice, """
0    1
1    2
dtype: int64
1    2
2    3
dtype: int64
0    1
1    2
dtype: int64
""")


@hipy.compiled_function
def fn_slice():
    s = pd.Series([1, 2, 3])
    print(s[:2])
    print(s[1:])
    print(s[:2])

def test_slice():
    check_prints(fn_slice, """
0    1
1    2
dtype: int64
1    2
2    3
dtype: int64
0    1
1    2
dtype: int64
""")

@hipy.compiled_function
def fn_const_lookup():
    s = pd.Series([1, 2, 3], index=pd.Index(['a', 'b', 'c'], name='index'))
    print(s['a'])
    print(s['b'])
    print(s['c'])
    s2 = pd.Series([1, "x", False], index=pd.Index(['a', 'b', 'c'], name='index'))
    print(s2['a'])
    print(s2['b'])
    print(s2['c'])

#    s2 = pd.Series(not_constant([1, 2, 3]), index=pd.Index(not_constant(['a', 'b', 'c']), name='index'))
#    print(s2['a'])
#    print(s2['b'])
#    print(s2['c'])
def test_const_lookup():
    check_prints(fn_const_lookup, """
1
2
3
1
x
False
""")

@hipy.compiled_function
def fn_lookup():
    s = pd.Series(not_constant([1, 2, 3]), index=pd.Index(not_constant(['a', 'b', 'c']), name='index'))
    print(s['a'])
    print(s['b'])
    print(s['c'])
    s2 = pd.Series(not_constant([1,2,3]))
    print(s2[0],intrinsics.isa(s2[0], np.int64))
    print(s2[1],intrinsics.isa(s2[1], np.int64))
    print(s2[2],intrinsics.isa(s2[2], np.int64))

def test_lookup():
    check_prints(fn_lookup, """
1
2
3
1 True
2 True
3 True
""")

@hipy.compiled_function
def fn_mask():
    s = pd.Series([1, 2, 3])
    print(s.mask(s > 1, 0))
    print(s.mask(s < 2, 0))
    print(s.mask(s == 2, 0))

def test_mask():
    check_prints(fn_mask, """
0    1
1    0
2    0
dtype: int64
0    0
1    2
2    3
dtype: int64
0    1
1    0
2    3
dtype: int64""")

@hipy.compiled_function
def fn_sum():
    s = pd.Series([1, 2, 3])
    print(s.sum())
    s = pd.Series([1.1, 2.2, 3.3])
    print(s.sum())
    s = pd.Series(["a", "b", "c"])
    print(s.sum())

def test_sum():
    check_prints(fn_sum, """
6
6.6
abc
""")

@hipy.compiled_function
def fn_numeric_ops():
    s = pd.Series([1, 2, 3])
    print(s + 1)
    print(s - 1)
    print(s * 2)
    print(s / 2)
    print(s.add(1))
    print(s.sub(1))
    print(s.mul(2))
    print(s.div(2))

def test_numeric_ops():
    check_prints(fn_numeric_ops, """
0    2
1    3
2    4
dtype: int64
0    0
1    1
2    2
dtype: int64
0    2
1    4
2    6
dtype: int64
0    0.5
1    1.0
2    1.5
dtype: float64
0    2
1    3
2    4
dtype: int64
0    0
1    1
2    2
dtype: int64
0    2
1    4
2    6
dtype: int64
0    0.5
1    1.0
2    1.5
dtype: float64
""")
