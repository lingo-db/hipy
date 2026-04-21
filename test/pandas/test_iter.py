import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.pandas
import hipy.lib.numpy
import numpy as np
import pandas as pd


@hipy.compiled_function
def fn_iter_series_int():
    s = pd.Series(not_constant([1, 2, 3]))
    total = 0
    for v in s:
        total = total + int(v)
    print(total)


def test_iter_series_int():
    check_prints(fn_iter_series_int, "6")


@hipy.compiled_function
def fn_iter_series_float():
    s = pd.Series(not_constant([1.5, 2.5, 3.0]))
    total = 0.0
    for v in s:
        total = total + float(v)
    print(total)


def test_iter_series_float():
    check_prints(fn_iter_series_float, "7.0")


@hipy.compiled_function
def fn_iter_series_str():
    s = pd.Series(not_constant(["a", "b", "c"]))
    joined = ""
    for v in s:
        joined = joined + v
    print(joined)


def test_iter_series_str():
    check_prints(fn_iter_series_str, "abc")


@hipy.compiled_function
def fn_iter_series_bool():
    s = pd.Series(not_constant([True, False, True, True]))
    count = 0
    for v in s:
        if v:
            count = count + 1
    print(count)


def test_iter_series_bool():
    check_prints(fn_iter_series_bool, "3")


@hipy.compiled_function
def fn_iter_series_const():
    # Series backed by _concrete_values (no not_constant) — materializes on iter.
    s = pd.Series([10, 20, 30])
    total = 0
    for v in s:
        total = total + int(v)
    print(total)


def test_iter_series_const():
    check_prints(fn_iter_series_const, "60")


@hipy.compiled_function
def fn_iter_series_build_list():
    s = pd.Series(not_constant([1, 2, 3, 4]))
    out = []
    for v in s:
        out.append(int(v) * 2)
    print(out)


def test_iter_series_build_list():
    check_prints(fn_iter_series_build_list, "[2, 4, 6, 8]")


@hipy.compiled_function
def fn_iter_df_columns():
    df = pd.DataFrame.from_dict({"a": not_constant([1, 2, 3]),
                                 "b": not_constant([4.0, 5.0, 6.0]),
                                 "c": not_constant(["x", "y", "z"])})
    for col in df:
        print(col)


def test_iter_df_columns():
    check_prints(fn_iter_df_columns, """
a
b
c
""")


@hipy.compiled_function
def fn_iter_df_columns_sum_int():
    # Iterate column names, then index into the frame to sum int columns.
    df = pd.DataFrame.from_dict({"a": [1, 2, 3],
                                 "b": [10, 20, 30],
                                 "c": [100, 200, 300]})
    grand_total = 0
    for col in df:
        grand_total = grand_total + int(df[col].sum())
    print(grand_total)


def test_iter_df_columns_sum_int():
    check_prints(fn_iter_df_columns_sum_int, "666")


@hipy.compiled_function
def fn_iter_range_index():
    df = pd.DataFrame.from_dict({"a": not_constant([10, 20, 30, 40])})
    total = 0
    for v in df.index:
        total = total + v
    print(total)


def test_iter_range_index():
    check_prints(fn_iter_range_index, "6")


@hipy.compiled_function
def fn_iter_index_str():
    idx = pd.Index(not_constant(["a", "b", "c"]), name="index")
    joined = ""
    for v in idx:
        joined = joined + v
    print(joined)


def test_iter_index_str():
    check_prints(fn_iter_index_str, "abc")


@hipy.compiled_function
def fn_iter_index_int():
    idx = pd.Index(not_constant([7, 8, 9]), name="index")
    total = 0
    for v in idx:
        total = total + int(v)
    print(total)


def test_iter_index_int():
    check_prints(fn_iter_index_int, "24")


@hipy.compiled_function
def fn_len_series():
    s = pd.Series(not_constant([1, 2, 3, 4, 5]))
    print(len(s))


def test_len_series():
    check_prints(fn_len_series, "5")


@hipy.compiled_function
def fn_len_df():
    df = pd.DataFrame.from_dict({"a": not_constant([1, 2, 3, 4])})
    print(len(df))


def test_len_df():
    check_prints(fn_len_df, "4")


@hipy.compiled_function
def fn_len_index():
    idx = pd.Index(not_constant(["a", "b", "c", "d", "e"]), name="index")
    print(len(idx))


def test_len_index():
    check_prints(fn_len_index, "5")


@hipy.compiled_function
def fn_len_range_index():
    df = pd.DataFrame.from_dict({"a": not_constant([1, 2, 3, 4, 5, 6, 7])})
    print(len(df.index))


def test_len_range_index():
    check_prints(fn_len_range_index, "7")


@hipy.compiled_function
def fn_iter_series_from_df():
    # Iterate a Series obtained via df[col] — exercises the parent-version path.
    df = pd.DataFrame.from_dict({"a": not_constant([1, 2, 3])})
    total = 0
    for v in df["a"]:
        total = total + int(v)
    print(total)


def test_iter_series_from_df():
    check_prints(fn_iter_series_from_df, "6")


@hipy.compiled_function
def fn_iter_df_then_series():
    # Outer loop over DataFrame iterates column names (unrolled at compile
    # time via __constiter__); inner loop iterates the selected Series
    # at runtime.
    df = pd.DataFrame.from_dict({"a": not_constant([1, 2, 3]),
                                 "b": not_constant([10, 20, 30])})
    total = 0
    for col in df:
        for v in df[col]:
            total = total + int(v)
    print(total)


def test_iter_df_then_series():
    check_prints(fn_iter_df_then_series, "66")


# df.iterrows() — yields (row_position, row) per row, matching pandas'
# shape for a default RangeIndex-backed frame. `row[col]` resolves a
# value by column name at IR time.

@hipy.compiled_function
def fn_iterrows_single_col_sum():
    df = pd.DataFrame.from_dict({"a": not_constant([1, 2, 3, 4])})
    total = 0
    for _, row in df.iterrows():
        total = total + int(row["a"])
    print(total)


def test_iterrows_single_col_sum():
    check_prints(fn_iterrows_single_col_sum, "10")


@hipy.compiled_function
def fn_iterrows_two_cols_product():
    df = pd.DataFrame.from_dict({"a": not_constant([1.0, 2.0, 3.0]),
                                 "b": not_constant([10.0, 20.0, 30.0])})
    total = 0.0
    for _, row in df.iterrows():
        total = total + float(row["a"]) * float(row["b"])
    print(total)


def test_iterrows_two_cols_product():
    # 1*10 + 2*20 + 3*30 = 140
    check_prints(fn_iterrows_two_cols_product, "140.0")


@hipy.compiled_function
def fn_iterrows_index_counter():
    # For a default RangeIndex frame, the yielded index should be 0..n-1.
    df = pd.DataFrame.from_dict({"a": not_constant([10, 20, 30])})
    total = 0
    for i, _ in df.iterrows():
        total = total + i
    print(total)


def test_iterrows_index_counter():
    check_prints(fn_iterrows_index_counter, "3")
