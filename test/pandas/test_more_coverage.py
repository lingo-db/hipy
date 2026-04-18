"""Additional pandas coverage — targets uncovered features in hipy/lib/pandas.

Before this file:
  - Series.fillna / reset_index / mask were not exercised.
  - DataFrame.fillna / reset_index / sort_values were not exercised.
  - pd.Timestamp(str) + pd.to_datetime(Series) were not exercised.
"""

import pandas as pd
import pytest

import hipy
import hipy.lib.pandas
import hipy.lib.numpy
import numpy as np
from hipy.interpreter import check_prints


@hipy.compiled_function
def fn_dataframe_reset_index():
    # DataFrame.reset_index() with a default RangeIndex — hipy/lib/pandas:693-701.
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    # reset_index promotes the current index into a column named "index"
    # (or "level_0" if "index" is already a column).
    print(df.reset_index())


def test_dataframe_reset_index():
    # HiPy's reset_index appends the new index column at the end and falls
    # back to "level_0" because the internal table already carries an "index"
    # column (from the base RangeIndex). That diverges from real pandas, which
    # puts the column first — captured here as the observed behavior.
    check_prints(fn_dataframe_reset_index, """
   a    b  level_0
0  1  0.1        0
1  2  0.2        1
2  3  0.3        2
""")


@hipy.compiled_function
def fn_series_reset_index():
    # Series.reset_index() → DataFrame with the Series as a column and the
    # previous index as a sibling column. hipy/lib/pandas:1138-1151.
    df = pd.DataFrame({"a": [10, 20, 30]})
    s = df["a"]
    print(s.reset_index())


def test_series_reset_index():
    # Same quirk as DataFrame.reset_index — new index column goes to the end
    # with name "level_0".
    check_prints(fn_series_reset_index, """
    a  level_0
0  10        0
1  20        1
2  30        2
""")


@hipy.compiled_function
def fn_series_fillna_float():
    # Series[float64].fillna(x) — hipy/lib/pandas:1131-1135. The native
    # df["a"] column is float64 because of NaN, and fillna fills those.
    df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
    filled = df["a"].fillna(0.0)
    print(filled)


def test_series_fillna_float():
    # HiPy's Series.fillna returns a new series without preserving .name —
    # so no "Name: a" line in the repr.
    check_prints(fn_series_fillna_float, """
0    1.0
1    0.0
2    3.0
dtype: float64
""")


@hipy.compiled_function
def fn_dataframe_sort_values_by_str():
    # DataFrame.sort_values("col") — hipy/lib/pandas:571-579. Single-string
    # 'by' goes through the `isa(by, str)` branch that wraps it in a list.
    df = pd.DataFrame({"k": [3, 1, 2], "v": ["c", "a", "b"]})
    print(df.sort_values("k"))


def test_dataframe_sort_values_by_str():
    check_prints(fn_dataframe_sort_values_by_str, """
   k  v
1  1  a
2  2  b
0  3  c
""")


@hipy.compiled_function
def fn_dataframe_sort_values_by_list_desc():
    # List-of-columns + ascending=False path exercises the ascending→list
    # broadcast in sort_values (line 576-577).
    df = pd.DataFrame({"k": [1, 2, 2, 1], "v": [20, 10, 30, 40]})
    print(df.sort_values(["k", "v"], ascending=False))


def test_dataframe_sort_values_by_list_desc():
    check_prints(fn_dataframe_sort_values_by_list_desc, """
   k   v
2  2  30
1  2  10
3  1  40
0  1  20
""")


@hipy.compiled_function
def fn_dataframe_fillna_float():
    # DataFrame.fillna walks every float column and forwards to Series.fillna.
    # hipy/lib/pandas:676-686.
    df = pd.DataFrame({
        "a": [1.0, float("nan"), 3.0],
        "b": [0.0, 1.0, float("nan")],
    })
    print(df.fillna(-1.0))


def test_dataframe_fillna_float():
    check_prints(fn_dataframe_fillna_float, """
     a    b
0  1.0  0.0
1 -1.0  1.0
2  3.0 -1.0
""")


@hipy.compiled_function
def fn_timestamp_from_str():
    # pd.Timestamp("…") — hipy/lib/pandas:1164-1165. Native path calls
    # date.parse on the string.
    ts = pd.Timestamp("2024-05-16")
    print(ts.year)


def test_timestamp_from_str():
    check_prints(fn_timestamp_from_str, """
2024
""")


@hipy.compiled_function
def fn_to_datetime_series():
    # pd.to_datetime(series) — hipy/lib/pandas:1192-1194. Maps Timestamp
    # over each element of a Series[int]. Printing .year proves each row
    # went through the Timestamp constructor.
    df = pd.DataFrame({"ns": [0, 1_700_000_000_000_000_000]})
    ts = pd.to_datetime(df["ns"])
    print(ts.apply(lambda t: t.year))


def test_to_datetime_series():
    check_prints(fn_to_datetime_series, """
0    1970
1    2023
dtype: int64
""")


@hipy.compiled_function
def fn_series_str_upper():
    # Series.str.upper → hipy/lib/pandas:783-784 (uncovered).
    df = pd.DataFrame.from_dict({"str": ['Abc', 'xY', 'z']})
    print(df['str'].str.upper())


def test_series_str_upper():
    check_prints(fn_series_str_upper, """
0    ABC
1     XY
2      Z
dtype: str""")


@hipy.compiled_function
def fn_series_dt_year():
    # Series.dt.year — hipy/lib/pandas:813-815 (_DateMethods.__hipy_getattr__).
    df = pd.DataFrame.from_dict({"d": [pd.Timestamp("2020-01-02"),
                                       pd.Timestamp("2024-12-31")]})
    print(df['d'].dt.year)


def test_series_dt_year():
    check_prints(fn_series_dt_year, """
0    2020
1    2024
dtype: int64
""")


@hipy.compiled_function
def fn_timestamp_hour():
    # Timestamp.hour — hipy/lib/pandas:1183-1184. Uses the date.get_hour builtin.
    ts = pd.Timestamp("2024-05-16")
    print(ts.hour)


def test_timestamp_hour():
    check_prints(fn_timestamp_hour, """
0
""")


@hipy.compiled_function
def fn_merge_left_right_index():
    # merge(..., left_index=True, right_index=True) — hipy/lib/pandas:727-732.
    # Joins on both frames' index columns instead of a `on=` column. We
    # construct the frames with the default RangeIndex so the join key is
    # the positional index.
    left = pd.DataFrame({"lv": [10, 20, 30]})
    right = pd.DataFrame({"rv": [100, 200, 300]})
    print(pd.merge(left, right, left_index=True, right_index=True))


def test_merge_left_right_index():
    check_prints(fn_merge_left_right_index, """
   lv   rv
0  10  100
1  20  200
2  30  300
""")
