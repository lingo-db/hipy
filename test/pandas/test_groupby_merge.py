"""Groupby/aggregate and merge tests for hipy's pandas layer.

Before this file, `DataFrameGroupBy.agg`, `merge(...)`, `table.join_inner`,
`table.join_left` and `table.aggregate` were not exercised in the test
suite. Each test below is targeted at one of those code paths.
"""

import pandas as _pd
import pandas as pd

import hipy
import hipy.lib.pandas
from hipy.interpreter import check_prints


@hipy.compiled_function
def fn_groupby_agg_dict():
    # dict form: agg({"col": "sum"}). Exercises handle_func("sum", int)
    # + table.aggregate + DataFrameGroupBy.agg dict branch.
    df = pd.DataFrame({"k": ["a", "b", "a", "b", "a"], "v": [1, 2, 3, 4, 5]})
    g = df.groupby(["k"])
    print(g.agg({"v": "sum"}))


def test_groupby_agg_dict():
    check_prints(fn_groupby_agg_dict, """
   v
k   
a  9
b  6
""")


@hipy.compiled_function
def fn_groupby_agg_min():
    # "min" path + float column.
    df = pd.DataFrame({"k": ["a", "b", "a", "b"], "v": [1.0, 5.0, 0.5, 2.5]})
    g = df.groupby(["k"])
    print(g.agg({"v": "min"}))


def test_groupby_agg_min():
    check_prints(fn_groupby_agg_min, """
     v
k     
a  0.5
b  2.5
""")


@hipy.compiled_function
def fn_groupby_agg_mean_kwargs():
    # kwargs form: agg(out=("col", "mean")).
    # Exercises the handle_func("mean", int) branch (returns tuple init
    # + tuple-valued agg), the kwargs loop in DataFrameGroupBy.agg, and
    # non-trivial finalize_fn (sum / count).
    df = pd.DataFrame({"k": ["a", "b", "a", "b", "a"], "v": [1, 2, 3, 4, 5]})
    g = df.groupby(["k"])
    print(g.agg(avg=("v", "mean")))


def test_groupby_agg_mean_kwargs():
    check_prints(fn_groupby_agg_mean_kwargs, """
   avg
k     
a  3.0
b  3.0
""")


@hipy.compiled_function
def fn_merge_inner_on():
    # pd.merge(..., on=...) inner join. Drives `merge` with `on` branch,
    # table.join_inner, and the post-join RangeIndex path.
    left = pd.DataFrame({"k": [1, 2, 3], "lv": [10, 20, 30]})
    right = pd.DataFrame({"k": [2, 3, 4], "rv": [200, 300, 400]})
    print(pd.merge(left, right, on="k"))


def test_merge_inner_on():
    check_prints(fn_merge_inner_on, """
   k  lv   rv
0  2  20  200
1  3  30  300
""")


@hipy.compiled_function
def fn_merge_inner_left_right_on():
    # Separate left_on/right_on branch in `merge`.
    left = pd.DataFrame({"lk": [1, 2, 3], "lv": [10, 20, 30]})
    right = pd.DataFrame({"rk": [2, 3, 4], "rv": [200, 300, 400]})
    print(pd.merge(left, right, left_on="lk", right_on="rk"))


def test_merge_inner_left_right_on():
    check_prints(fn_merge_inner_left_right_on, """
   lk  lv  rk   rv
0   2  20   2  200
1   3  30   3  300
""")


@hipy.compiled_function
def fn_merge_left():
    # how="left": right-side int column gets cast to float64 so NULL
    # rows survive the join; hits table.join_left.
    left = pd.DataFrame({"k": [1, 2, 3], "lv": [10, 20, 30]})
    right = pd.DataFrame({"k": [2, 3, 4], "rv": [200, 300, 400]})
    print(pd.merge(left, right, on="k", how="left"))


def test_merge_left():
    check_prints(fn_merge_left, """
   k  lv     rv
0  1  10    NaN
1  2  20  200.0
2  3  30  300.0
""")


@hipy.compiled_function
def fn_groupby_series_nunique():
    # df.groupby(by)[col].nunique() — hits DataFrameGroupBySeriesGroupBy.nunique,
    # which does two back-to-back table.aggregate calls.
    df = pd.DataFrame({
        "k": ["a", "a", "a", "b", "b"],
        "v": [1, 1, 2, 3, 3],
    })
    print(df.groupby(["k"])["v"].nunique())


def test_groupby_series_nunique():
    check_prints(fn_groupby_series_nunique, """
k
a    2
b    1
Name: v, dtype: int64
""")
