"""Tests for UDFs from the QURE Benchmark scalar suite.

Source: ``/home/michael/Downloads/QURE Benchmark/scalar/q*.py``.

Each source file wraps ``udf_0`` as a PySpark ``pandas_udf`` taking one
or more ``pd.Series`` arguments. The bodies are copied here verbatim
into ``@hipy.compiled_function`` wrappers (dropping the pyspark
imports/decorators, and adding inner imports that the original forgot
— e.g. the top-level ``np`` used by q11).

Each wrapper is called with ``pd.Series(not_constant([...]))`` inputs
and the result Series is iterated to stdout so that value comparisons
don't have to thread HiPy's float-to-string precision through pandas'
display formatting.

UDFs that exercise features the C++ backend/shims don't yet support
(``np.minimum``/``np.power``/``np.linspace``, ``Series.quantile``,
``Series.mean``/``.std``/``.map``/``.abs``, ``zip``, generator
``yield``, row/struct inputs) are marked ``pytest.mark.xfail`` rather
than silently dropped, so the set of missing features remains visible.
"""
import pytest

import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.pandas
import pandas as pd
import hipy.lib.numpy
import numpy as np


# ---------------------------------------------------------------------------
# q1 — np.minimum / np.abs / np.sqrt / np.power over pandas Series
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q1(dfs1, dfs2, dfs3, dfs4):
    import numpy as np
    import pandas as pd
    rs = np.minimum(np.abs(dfs1 - dfs3), np.abs(dfs1 - dfs3 - 360))
    res = np.sqrt(np.power(rs, 2) + np.power(dfs4 - dfs2, 2))
    return pd.Series(res)


@hipy.compiled_function
def fn_q1():
    s1 = pd.Series(not_constant([10, 20, 30]))
    s2 = pd.Series(not_constant([1, 2, 3]))
    s3 = pd.Series(not_constant([4, 5, 6]))
    s4 = pd.Series(not_constant([7, 8, 9]))
    for v in udf_q1(s1, s2, s3, s4):
        print(v)


@pytest.mark.xfail(reason="numpy shim lacks minimum/abs/power over Series")
def test_q1():
    check_prints(fn_q1, "")


# ---------------------------------------------------------------------------
# q2 — elementwise sigmoid via Python for-loop + np.exp(scalar)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q2(l_extendedprice):
    transformed_values = []

    # Apply sigmoid scaling to each value
    for price in l_extendedprice:
        sigmoid_value = 1 / (1 + np.exp(-price))
        transformed_values.append(sigmoid_value)

    # Return the sigmoid transformed values
    return pd.Series(transformed_values)


@hipy.compiled_function
def fn_q2():
    s = pd.Series(not_constant([0.0]))
    for v in udf_q2(s):
        print(v)


@pytest.mark.xfail(reason="numpy float64 scalar has no __neg__; -price inside the loop fails")
def test_q2():
    check_prints(fn_q2, """
0.5
""")


# ---------------------------------------------------------------------------
# q3 — Series.quantile + apply(axis=1)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q3(quantity):
    Q1 = quantity.quantile(0.25)
    Q3 = quantity.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier =  quantity.apply(lambda x: x < lower_bound or x > upper_bound, axis = 1)
    return outlier


@hipy.compiled_function
def fn_q3():
    s = pd.Series(not_constant([1.0, 2.0, 3.0, 100.0]))
    for v in udf_q3(s):
        print(v)


@pytest.mark.xfail(reason="pandas shim lacks Series.quantile and apply(axis=1)")
def test_q3():
    check_prints(fn_q3, "")


# ---------------------------------------------------------------------------
# q4 — np.exp / arithmetic directly on a Series
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q4(dfs):
    import numpy as np
    return np.exp(dfs / 50) + 10


@hipy.compiled_function
def fn_q4():
    s = pd.Series(not_constant([0.0, 50.0]))
    for v in udf_q4(s):
        print(v)


@pytest.mark.xfail(reason="numpy shim doesn't dispatch exp over a Series")
def test_q4():
    check_prints(fn_q4, "")


# ---------------------------------------------------------------------------
# q5 — zip over two Series
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q5(l_extendedprice, l_shipmode):
    discount_rates = {
        'AIR': 0.08, 'MAIL': 0.09, 'OTHER': 0.07    }

    results = []
    for price, mode in zip(l_extendedprice, l_shipmode):
        if price < 0.0:
            result_summary = "Invalid Price"
        else:
            tax_rate = discount_rates.get(mode, 0.0)
            tax_amount =  price * tax_rate
            total_price = price + tax_amount

            result_summary = "Mode: " + mode + ", Extended Price: $" + str(price) + ", Discount Rate: " + str(tax_rate*100.0) + "%, Tax Amount: $"+ str(tax_amount) + ", Total Price: $ " + str(total_price)
        results.append(result_summary)

    return pd.Series(results)


@hipy.compiled_function
def fn_q5():
    prices = pd.Series(not_constant([100.0, -1.0]))
    modes = pd.Series(not_constant(["AIR", "MAIL"]))
    for v in udf_q5(prices, modes):
        print(v)


@pytest.mark.xfail(reason="HiPy doesn't support the built-in zip()")
def test_q5():
    check_prints(fn_q5, "")


# ---------------------------------------------------------------------------
# q6 — nested helper + list-comp over a Series
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q6(dfs):
    # Original body had a function-scope ``import pandas as pd``; HiPy
    # forbids inner ``import`` statements inside compiled functions, so
    # we rely on the module-level import instead.
    def get_idx(val):
        mp = {"MAIL": 1, "AIR": 2, "OTHER": 9999}
        if val not in mp:
            val = "OTHER"
        return mp[val]

    return pd.Series([get_idx(v) for v in dfs])


@hipy.compiled_function
def fn_q6():
    s = pd.Series(not_constant(["MAIL", "AIR", "TRUCK"]))
    for v in udf_q6(s):
        print(v)


@pytest.mark.xfail(reason="Nested function calling `val not in mp` through a list-comp lambda trips HiPy's bool_not path")
def test_q6():
    check_prints(fn_q6, """
1
2
9999
""")


# ---------------------------------------------------------------------------
# q7 — Series.map(...)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q7(series):
    mapping = {
        "AIR": "0",
        "MAIL": "1",
        "RAIL": "2",
        "SHIP": "3",
        "TRUCK": "4",
        "REG AIR": "5",
        "FOB": "6",
    }

    transformed_series = series.map(mapping)
    result_series = transformed_series + "-" + series
    return result_series


@hipy.compiled_function
def fn_q7():
    s = pd.Series(not_constant(["AIR", "MAIL", "RAIL"]))
    for v in udf_q7(s):
        print(v)


@pytest.mark.xfail(reason="pandas shim lacks Series.map")
def test_q7():
    check_prints(fn_q7, "")


# ---------------------------------------------------------------------------
# q8 — for-loop + np.log(scalar) with branches
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q8(s_acctbal):
    transformed_values = []

    # Apply a logarithmic transformation with a custom shift for different ranges
    for bal in s_acctbal:
        if bal < 10000:
            log_value = np.log(bal + 10)  # Add a shift to avoid log(0)
        elif bal < 50000:
            log_value = np.log(bal + 100)  # Larger shift for mid-range values
        else:
            log_value = np.log(bal + 1000)  # Larger shift for large balances
        transformed_values.append(log_value)

    # Return the custom logarithmic transformed values
    return pd.Series(transformed_values)


@hipy.compiled_function
def fn_q8():
    # Inputs chosen so that np.log(…) round-trips to the exact 6-digit
    # string HiPy emits via builtin::float_to_string.
    s = pd.Series(not_constant([0.0, 20000.0, 100000.0]))
    for v in udf_q8(s):
        print(v)


def test_q8():
    check_prints(fn_q8, """
2.302585
9.908475
11.522876
""")


# ---------------------------------------------------------------------------
# q9 — Series.add / Series.abs arithmetic
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q9(dfs1, dfs2, dfs3, dfs4):
    diff1 = dfs1.add(-dfs3)
    diff11 = diff1.abs()
    diff2 = dfs2.add(-dfs4)
    diff21 = diff2.abs()
    return diff11.add(diff21)


@hipy.compiled_function
def fn_q9():
    s1 = pd.Series(not_constant([10, 20]))
    s2 = pd.Series(not_constant([5, 15]))
    s3 = pd.Series(not_constant([3, 4]))
    s4 = pd.Series(not_constant([1, 2]))
    for v in udf_q9(s1, s2, s3, s4):
        print(v)


@pytest.mark.xfail(reason="pandas shim lacks Series.abs / unary minus on Series")
def test_q9():
    check_prints(fn_q9, "")


# ---------------------------------------------------------------------------
# q10 — list-comp over Series
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q10(dfs):
    # Dropped the UDF-scope ``import pandas as pd`` (HiPy rejects nested
    # imports); the module-level ``pd`` is in scope.
    return pd.Series([(i - 10) / (20 - 10) for i in dfs])


@hipy.compiled_function
def fn_q10():
    s = pd.Series(not_constant([10.0, 20.0, 30.0]))
    for v in udf_q10(s):
        print(v)


def test_q10():
    check_prints(fn_q10, """
0.0
1.0
2.0
""")


# ---------------------------------------------------------------------------
# q11 — numpy masking, np.empty, np.logical_*, broadcasting
# The original source references ``np`` without importing it inside the
# UDF body. That's preserved as an xfail regardless.
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q11(dfs1, dfs2, dfs3, dfs4):
    import sys
    import math
    import pandas as pd
    import numpy as np  # <-- the original forgets this

    MAX_ANGLE = 2

    f = np.logical_or(np.logical_and(np.abs(dfs1 - dfs3) > MAX_ANGLE, np.abs(dfs1 - dfs3 - 360) > MAX_ANGLE),
                      np.abs(dfs2 - dfs4) > MAX_ANGLE)
    res = np.empty(len(dfs1))

    res[f] = sys.float_info.max

    # the opposite mask (these rows will be calculated)
    nf = np.logical_not(f)

    # Gnomonic projection centered on dfs1, dec1
    ra1 = dfs1[nf] / 180 * math.pi
    ra2 = dfs3[nf] / 180 * math.pi
    dec1 = dfs2[nf] / 180 * math.pi
    dec2 = dfs4[nf] / 180 * math.pi
    ra = np.minimum(np.abs(ra1 - ra2), 360 - np.abs(ra1 - ra2))
    dec = np.abs(dec1 - dec2)
    cosc = np.cos(ra) * np.cos(dec)
    cosdec = np.cos(dec)
    res[nf] = np.sqrt(np.power(cosdec, 2) * np.power(np.sin(ra), 2) +
                      np.power(np.sin(dec), 2)) / cosc
    return pd.Series(res)


@hipy.compiled_function
def fn_q11():
    s1 = pd.Series(not_constant([10.0]))
    s2 = pd.Series(not_constant([20.0]))
    s3 = pd.Series(not_constant([15.0]))
    s4 = pd.Series(not_constant([25.0]))
    for v in udf_q11(s1, s2, s3, s4):
        print(v)


@pytest.mark.xfail(reason="Needs np.empty/logical_or/logical_and/logical_not/minimum/power + boolean-mask indexing")
def test_q11():
    check_prints(fn_q11, "")


# ---------------------------------------------------------------------------
# q12 — elementwise cubic polynomial
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q12(o_totalprice):
    transformed_values = []

    # Apply a cubic transformation for each value
    for price in o_totalprice:
        cubic_value = price ** 3 - 2 * (price ** 2) + 5 * price - 7
        transformed_values.append(cubic_value)

    # Return the cubic transformed values as a Pandas Series
    return pd.Series(transformed_values)


@hipy.compiled_function
def fn_q12():
    s = pd.Series(not_constant([0.0, 1.0, 2.0]))
    for v in udf_q12(s):
        print(v)


def test_q12():
    check_prints(fn_q12, """
-7.0
-3.0
3.0
""")


# ---------------------------------------------------------------------------
# q13 — for-loop with string methods
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q13(l_comment):
    results = []
    for comment in l_comment:
        cleaned_comment1 = comment.replace('!', '').replace('@', '')
        cleaned_comment2 = cleaned_comment1.replace('#', '').lower()
        word_count = len(cleaned_comment2.split(" "))
        if 'urgent' in cleaned_comment2 or 'important' in cleaned_comment2:
            flag = "Urgent"
        else:
            flag = "Normal"
        result = "Cleaned Comment: " + cleaned_comment2 + ", Word Count: " + str(word_count) + ", Flag: " + flag
        results.append(result)

    return pd.Series(results)


@hipy.compiled_function
def fn_q13():
    s = pd.Series(not_constant(["URGENT!! Ship now", "Regular delivery", "IMPORTANT@note#"]))
    for v in udf_q13(s):
        print(v)


def test_q13():
    check_prints(fn_q13, """
Cleaned Comment: urgent ship now, Word Count: 3, Flag: Urgent
Cleaned Comment: regular delivery, Word Count: 2, Flag: Normal
Cleaned Comment: importantnote, Word Count: 1, Flag: Urgent
""")


# ---------------------------------------------------------------------------
# q14 — np.linspace
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q14(dfs):
    import pandas as pd
    import numpy as np
    return pd.Series(np.linspace(0, 1, len(dfs)))


@hipy.compiled_function
def fn_q14():
    s = pd.Series(not_constant([1.0, 2.0, 3.0, 4.0]))
    for v in udf_q14(s):
        print(v)


@pytest.mark.xfail(reason="numpy shim lacks np.linspace")
def test_q14():
    check_prints(fn_q14, "")


# ---------------------------------------------------------------------------
# q15 — struct / row-dict input
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q15(row):
    base_price = row['l_extendedprice']
    price_after_discount = base_price * (1 - row['l_discount'])
    price_after_tax = price_after_discount * (1 + row['l_tax'])

    if row['l_returnflag'] == 'N':
        if row['l_linestatus'] == 'O':
            seasonal_factor = 0.95
        else:
            seasonal_factor = 1.05
    else:
        seasonal_factor = 1.00

    seasonally_adjusted_price = price_after_tax * seasonal_factor

    if row['l_quantity'] < 10:
        stock_adjustment_factor = 1.1
    elif row['l_quantity'] < 20:
        stock_adjustment_factor = 1.05
    else:
        stock_adjustment_factor = 1.00

    final_price = seasonally_adjusted_price * stock_adjustment_factor

    if row['l_partkey'] in [155, 777]:
        final_price *= 1.02

    return final_price


@hipy.compiled_function
def fn_q15():
    row = {
        'l_extendedprice': not_constant(100.0),
        'l_discount': not_constant(0.1),
        'l_tax': not_constant(0.05),
        'l_returnflag': not_constant('N'),
        'l_linestatus': not_constant('O'),
        'l_quantity': not_constant(5),
        'l_partkey': not_constant(155),
    }
    print(udf_q15(row))


@pytest.mark.xfail(reason="HiPy doesn't support heterogeneous dict-indexed row records the way Spark does")
def test_q15():
    check_prints(fn_q15, "")


# ---------------------------------------------------------------------------
# q16 — generator UDF (``yield``)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q16(dfs_iter):
    for dfs in dfs_iter:
        yield dfs + 2


@hipy.compiled_function
def fn_q16():
    s = pd.Series(not_constant([1, 2, 3]))
    for batch in udf_q16([s]):
        for v in batch:
            print(v)


@pytest.mark.xfail(reason="HiPy doesn't compile generator (``yield``) functions")
def test_q16():
    check_prints(fn_q16, "")


# ---------------------------------------------------------------------------
# q17 — zip over three Series (output template has literal ``{price}`` — the
# original UDF forgot the leading ``f``, so the braces survive verbatim)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q17(l_extendedprice, l_discount, l_quantity):
    results = []
    for price, discount, quantity in zip(l_extendedprice, l_discount, l_quantity):
        final_price = price * (1 - discount)
        if quantity > 50:
            final_price = final_price * 0.9
        if discount < 0.05:
            discount_level = "Low"
        elif discount < 0.15:
            discount_level = "Medium"
        else:
            discount_level = "High"
        r = "Original Price: {price}, Final Price: {final_price}, Discount Level: {discount_level}"
        results.append(r)

    return pd.Series(results)


@hipy.compiled_function
def fn_q17():
    p = pd.Series(not_constant([100.0]))
    d = pd.Series(not_constant([0.1]))
    q = pd.Series(not_constant([5.0]))
    for v in udf_q17(p, d, q):
        print(v)


@pytest.mark.xfail(reason="HiPy doesn't support the built-in zip()")
def test_q17():
    check_prints(fn_q17, "")


# ---------------------------------------------------------------------------
# q18 — redact / sentiment for-loop
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q18(l_comment):
    sensitive_words = ['confidential', 'urgent', 'damaged']
    positive_keywords = ['happy', 'pleased', 'good']
    negative_keywords = ['bad', 'unhappy', 'dissatisfied']

    results = []
    for comment in l_comment:
        no_comments = len(comment)
        if no_comments == 0:
            result_summary = "No comment provided"
        else:
            for word in sensitive_words:
                comment = comment.replace(word, '*****')
            comment_lower = comment.lower()
            positive_count = sum(comment_lower.count(word) for word in positive_keywords)
            negative_count = sum(comment_lower.count(word) for word in negative_keywords)

            if positive_count > negative_count:
                sentiment = "Positive"
            else:
                if negative_count > positive_count:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

            result_summary = "Redacted Comment: " + comment + ", Sentiment: " + sentiment + ", Positive Count: " + str(positive_count) + ", Negative Count: " + str(negative_count)

        results.append(result_summary)

    return pd.Series(results)


@hipy.compiled_function
def fn_q18():
    s = pd.Series(not_constant(["", "good happy day", "bad urgent situation"]))
    for v in udf_q18(s):
        print(v)


def test_q18():
    check_prints(fn_q18, """
No comment provided
Redacted Comment: good happy day, Sentiment: Positive, Positive Count: 2, Negative Count: 0
Redacted Comment: bad ***** situation, Sentiment: Negative, Positive Count: 0, Negative Count: 1
""")


# ---------------------------------------------------------------------------
# q19 — Series.mean() / Series.std()
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q19(dfs):
    return (dfs - dfs.mean()) / dfs.std()


@hipy.compiled_function
def fn_q19():
    s = pd.Series(not_constant([1.0, 2.0, 3.0, 4.0]))
    for v in udf_q19(s):
        print(v)


@pytest.mark.xfail(reason="pandas shim lacks Series.mean / Series.std")
def test_q19():
    check_prints(fn_q19, "")
