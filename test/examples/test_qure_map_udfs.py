"""Tests for UDFs from the QURE Benchmark map suite.

Source: ``/home/michael/Downloads/QURE Benchmark/map/q*.py``.

Each source file wraps ``udf_0`` as a PySpark ``pandas_udf`` of type
``GROUPED_MAP`` that takes a whole DataFrame (or, for q11, two
Series) and returns a DataFrame. The bodies are copied here verbatim
into ``@hipy.compiled_function`` wrappers (dropping the pyspark
imports/decorators).

Each test calls the UDF with a small lineitem-like DataFrame (or
orders DataFrame for q8) constructed with ``not_constant(...)`` and
prints the returned DataFrame for ``check_prints``.

UDFs that exercise features the C++ backend/shims don't yet support
(``df.iterrows`` + row access, ``pd.to_numeric``/``df.loc``
row-conditional assignment, ``np.diff``/``np.absolute``, date
comparisons on columns, ``Series.min``/``.max``, ``pd.merge`` inside
a UDF, …) are marked ``pytest.mark.xfail`` rather than silently
dropped, so the set of missing features remains visible.
"""
import pytest

import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.pandas
import pandas as pd
import hipy.lib.numpy
import numpy as np


@hipy.compiled_function
def _lineitem_df():
    # Small lineitem fixture shared by most UDFs. ``not_constant`` defeats
    # constant folding so the computed UDF body actually emits IR.
    return pd.DataFrame({
        "l_orderkey": not_constant([1, 2, 3]),
        "l_partkey": not_constant([100, 200, 300]),
        "l_suppkey": not_constant([10, 20, 30]),
        "l_linenumber": not_constant([1, 2, 3]),
        "l_quantity": not_constant([10.0, 20.0, 30.0]),
        "l_extendedprice": not_constant([1000.0, 5000.0, 15000.0]),
        "l_discount": not_constant([0.1, 0.2, 0.3]),
        "l_tax": not_constant([0.1, 0.2, 0.3]),
        "l_returnflag": not_constant(["N", "R", "A"]),
        "l_linestatus": not_constant(["O", "F", "F"]),
        "l_shipmode": not_constant(["AIR", "MAIL", "TRUCK"]),
        "l_comment": not_constant(["fast", "slow", "on time"]),
    })


# ---------------------------------------------------------------------------
# q1 — piecewise-linear transform over l_extendedprice
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q1(df):
    l_extendedprice = df["l_extendedprice"]
    transformed_values = []
    for price in l_extendedprice:
        if price < 5000.0:
            transformed_values.append(price * 1.1)
        else:
           if (5000.0 <= price  and  price  < 10000.0):
                transformed_values.append(5000.0 * 1.1 + (price - 5000.0) * 0.9)
           else:
                transformed_values.append(5000.0 * 1.1 + 5000.0 * 0.9 + (price - 10000.0) * 0.7)
    return pd.DataFrame({"transformed_price": transformed_values})


@hipy.compiled_function
def fn_q1():
    df = _lineitem_df()
    print(udf_q1(df))


def test_q1():
    # 1000 → 1100, 5000 → 5500, 15000 → 5000*1.1 + 5000*0.9 + 5000*0.7 = 13500.
    check_prints(fn_q1, """
   transformed_price
0             1100.0
1             5500.0
2            13500.0
""")


# ---------------------------------------------------------------------------
# q2 — pd.to_numeric + column arithmetic
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q2(df):
    df['l_extendedprice'] = pd.to_numeric(df['l_extendedprice'], errors='coerce').fillna(0)
    df['l_discount'] = pd.to_numeric(df['l_discount'], errors='coerce').fillna(0)
    df['l_tax'] = pd.to_numeric(df['l_tax'], errors='coerce').fillna(0)
    df['l_quantity'] = pd.to_numeric(df['l_quantity'], errors='coerce').fillna(0)

    df['discounted_price'] = df['l_extendedprice'] * (1.0 - df['l_discount'])
    df['total_cost'] = df['discounted_price'] * (1.0 + df['l_tax'])
    df['profit'] = df['total_cost'] - df['l_extendedprice']
    df['profit_margin'] = df['profit'] / df['total_cost']

    return df[['discounted_price', 'total_cost', 'profit', 'profit_margin']]


@hipy.compiled_function
def fn_q2():
    df = _lineitem_df()
    print(udf_q2(df))


@pytest.mark.xfail(reason="pd.to_numeric is not implemented in the pandas shim")
def test_q2():
    check_prints(fn_q2, "")


# ---------------------------------------------------------------------------
# q3 — df.loc row-conditional assignment + pd.to_numeric
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q3(df):
    import pandas as pd
    df['l_extendedprice1'] = pd.to_numeric(df['l_extendedprice'], errors='coerce').fillna(0)
    df['l_discount1'] = pd.to_numeric(df['l_discount'], errors='coerce').fillna(0)
    df['price1'] = df['l_extendedprice'] * (1 - df['l_discount'])
    df['description'] = 'Error'
    df.loc[df['price1'] > 0, 'description'] = 'Reasonable Price'
    df.loc[df['price1'] > 1000, 'description'] = 'High Price'
    return df[['l_extendedprice1', 'l_discount1', 'price1', 'description']]


@hipy.compiled_function
def fn_q3():
    df = _lineitem_df()
    print(udf_q3(df))


@pytest.mark.xfail(reason="pd.to_numeric and df.loc row-conditional assignment are not supported")
def test_q3():
    check_prints(fn_q3, "")


# ---------------------------------------------------------------------------
# q4 — np.diff + np.std + np.absolute
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q4(df):
    import numpy as np
    l_quantity_diff = np.diff(df["l_quantity"])
    l_quantity_stddev_diff = np.std(np.diff(l_quantity_diff))
    l_quantity_stddev = np.std(l_quantity_diff)
    variation = np.sqrt(np.absolute((2 * l_quantity_stddev ** 2) - (0.5 * l_quantity_stddev_diff ** 2)))
    import pandas as pd
    return pd.DataFrame({"l_value": [variation]})


@hipy.compiled_function
def fn_q4():
    df = _lineitem_df()
    print(udf_q4(df))


def test_q4():
    # quantity=[10,20,30] → diff=[10,10] → diff-of-diff=[0].
    # stddev(diff)=0, stddev(diff-of-diff)=0 → variation=sqrt(|0-0|)=0.
    check_prints(fn_q4, """
   l_value
0      0.0
""")


# ---------------------------------------------------------------------------
# q5 — boolean list-comps over Series columns
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q5(df):
    df["l_quantity1"] = [True if (x > 25 or x < 10) else False for x in df["l_quantity"]]
    df["l_discount1"] = [True if (y < 0.7 and y > 0.2) else False for y in df["l_discount"]]
    return df[["l_quantity1", "l_discount1"]]


@hipy.compiled_function
def fn_q5():
    df = _lineitem_df()
    print(udf_q5(df))


@pytest.mark.xfail(reason="Boolean list-comp element-assignment to a DataFrame column is not supported")
def test_q5():
    check_prints(fn_q5, "")


# ---------------------------------------------------------------------------
# q6 — per-column standardisation, returns a multi-column frame
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q6(df):
    l_quantity_col = df["l_quantity"]
    l_discount_col = df["l_discount"]
    l_tax_col = df["l_tax"]
    l_quantity1 = (df["l_quantity"] - df["l_quantity"].mean()) / df["l_quantity"].std()
    l_discount1 = (df["l_discount"] - df["l_discount"].mean()) / df["l_discount"].std()
    l_tax1 = (df["l_tax"] - df["l_tax"].mean()) / df["l_tax"].std()
    result = pd.DataFrame({
        "l_linenumber": df["l_linenumber"],
        "l_quantity1": l_quantity1,
        "l_discount1": l_discount1,
        "l_tax1": l_tax1
    })
    return result


@hipy.compiled_function
def fn_q6():
    df = _lineitem_df()
    print(udf_q6(df))


def test_q6():
    # quantity=[10,20,30] lands on exact floats, so its standardisation is
    # [-1.0, 0.0, 1.0]. discount/tax=[0.1,0.2,0.3] accumulate a double
    # rounding error of ~2.78e-16 in the mean, so pandas switches to
    # scientific notation — match pandas' own output.
    check_prints(fn_q6, """
   l_linenumber  l_quantity1   l_discount1        l_tax1
0             1         -1.0 -1.000000e+00 -1.000000e+00
1             2          0.0 -2.775558e-16 -2.775558e-16
2             3          1.0  1.000000e+00  1.000000e+00
""")


# ---------------------------------------------------------------------------
# q7 — pd.merge inside a UDF
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q7(df):
    import pandas as pd
    rdf = pd.DataFrame({
        "l_shipmode": ["AIR", "MAIL", "RAIL", "SHIP", "TRUCK", "REG AIR", "FOB"],
        "l_shipmodecode": [0, 1, 2, 3, 4, 5, 6],
    })
    df1 = pd.merge(df, rdf, how="inner", on="l_shipmode")
    return df1[["l_linenumber","l_shipmodecode"]]


@hipy.compiled_function
def fn_q7():
    df = _lineitem_df()
    print(udf_q7(df))


@pytest.mark.xfail(reason="pd.merge inside a compiled function is not currently supported")
def test_q7():
    check_prints(fn_q7, "")


# ---------------------------------------------------------------------------
# q8 — cumulative sum with a hard cap (orders.o_totalprice)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q8(df):
    o_totalprice = df["o_totalprice"]
    threshold = 0.5
    cumulative_sum = 0.0
    cum_sum_values = []
    for price in o_totalprice:
        cumulative_sum += price
        if cumulative_sum > threshold:
            cum_sum_values.append(threshold)
        else:
            cum_sum_values.append(cumulative_sum)
    return pd.DataFrame({"cumulative_sum_with_threshold": cum_sum_values})


@hipy.compiled_function
def fn_q8():
    df = pd.DataFrame({"o_totalprice": not_constant([0.1, 0.3, 0.5])})
    print(udf_q8(df))


def test_q8():
    # cumsum = [0.1, 0.4, 0.9]; with threshold=0.5 → [0.1, 0.4, 0.5].
    check_prints(fn_q8, """
   cumulative_sum_with_threshold
0                            0.1
1                            0.4
2                            0.5
""")


# ---------------------------------------------------------------------------
# q9 — df.loc row-conditional labels (string)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q9(df):
    df['l_extendedprice1'] = pd.to_numeric(df['l_extendedprice'], errors='coerce').fillna(0)
    df['l_discount1'] = pd.to_numeric(df['l_discount'], errors='coerce').fillna(0)
    df['l_profit'] = df['l_extendedprice'] * (1 - df['l_discount'])
    df['l_profitability'] = 'Loss'
    df.loc[df['l_profit'] > 0, 'l_profitability'] = 'Profit'
    df.loc[df['l_profit'] > 1000, 'l_profitability'] = 'High Profit'
    return df[['l_extendedprice1', 'l_discount1', 'l_profit', 'l_profitability']]


@hipy.compiled_function
def fn_q9():
    df = _lineitem_df()
    print(udf_q9(df))


@pytest.mark.xfail(reason="pd.to_numeric and df.loc row-conditional assignment are not supported")
def test_q9():
    check_prints(fn_q9, "")


# ---------------------------------------------------------------------------
# q10 — df.iterrows scalar reduction
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q10(df):
    taxmax = -1
    for _, row in df.iterrows():
        taxmax = max(
            taxmax,
            row["l_quantity"] * row["l_discount"] * row["l_tax"]
        )
    import pandas as pd
    return pd.DataFrame({"l_taxmax": [taxmax]})


@hipy.compiled_function
def fn_q10():
    df = _lineitem_df()
    print(udf_q10(df))


@pytest.mark.xfail(reason="df.iterrows with per-row column lookup is not supported")
def test_q10():
    check_prints(fn_q10, "")


# ---------------------------------------------------------------------------
# q11 — takes two Series (not a df), zip + substring check
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q11(l_comment, l_shipmode):
    res = []
    for ls, ns in zip(l_comment,  l_shipmode):
        s = ls + ns
        r = "AIR" in s
        res.append(r)
    l_udf = pd.DataFrame({"udf_col": res})
    return l_udf


@hipy.compiled_function
def fn_q11():
    c = pd.Series(not_constant(["fast", "slow", "on time"]))
    m = pd.Series(not_constant(["AIR", "MAIL", "TRUCK"]))
    print(udf_q11(c, m))


def test_q11():
    # "fast"+"AIR" contains AIR; other two don't.
    check_prints(fn_q11, """
   udf_col
0     True
1    False
2    False
""")


# ---------------------------------------------------------------------------
# q12 — df.iterrows building a list
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q12(df):
    taxmax_list = []
    for _, row in df.iterrows():
        taxmax = max(max(row["l_quantity"], row["l_discount"] * 100), row["l_tax"] * 100)
        taxmax_list.append(taxmax)
    import pandas as pd
    return pd.DataFrame({"l_taxmax": taxmax_list})


@hipy.compiled_function
def fn_q12():
    df = _lineitem_df()
    print(udf_q12(df))


@pytest.mark.xfail(reason="df.iterrows with per-row column lookup is not supported")
def test_q12():
    check_prints(fn_q12, "")


# ---------------------------------------------------------------------------
# q13 — Series.min over a date column + boolean filter on a DataFrame
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q13(df):
    oldest_date = df["l_commitdate"].min()
    df = df[df["l_shipdate"] >= oldest_date]
    return df


@hipy.compiled_function
def fn_q13():
    df = pd.DataFrame({
        "l_linenumber": not_constant([1, 2, 3]),
        "l_shipdate": not_constant(["2024-01-10", "2024-02-10", "2024-03-10"]),
        "l_commitdate": not_constant(["2024-01-05", "2024-02-15", "2024-03-01"]),
    })
    print(udf_q13(df))


@pytest.mark.xfail(reason="Series.min on a date/string column is not supported")
def test_q13():
    check_prints(fn_q13, "")


# ---------------------------------------------------------------------------
# q14 — min-max scaling (needs Series.min / Series.max)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q14(df):
    l_quantity = df["l_quantity"]
    min_val = l_quantity.min()
    max_val = l_quantity.max()

    scaled_result = []
    for value in l_quantity:
        scaled_result.append((value - min_val) / (max_val - min_val))

    return pd.DataFrame({"scaled_quantity": scaled_result})


@hipy.compiled_function
def fn_q14():
    df = _lineitem_df()
    print(udf_q14(df))


def test_q14():
    # quantity=[10,20,30] → (value-10)/20 → [0.0, 0.5, 1.0].
    check_prints(fn_q14, """
   scaled_quantity
0              0.0
1              0.5
2              1.0
""")


# ---------------------------------------------------------------------------
# q15 — single-column standardisation
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q15(df):
    l_quantity_col = df["l_quantity"]
    standardized_quantity = (l_quantity_col - l_quantity_col.mean()) / l_quantity_col.std()
    result = pd.DataFrame({
        "standardized_quantity": standardized_quantity
    })
    return result


@hipy.compiled_function
def fn_q15():
    df = _lineitem_df()
    print(udf_q15(df))


def test_q15():
    # quantity=[10,20,30] → mean=20, std=10 → [-1, 0, 1].
    check_prints(fn_q15, """
   standardized_quantity
0                   -1.0
1                    0.0
2                    1.0
""")
