"""Tests for UDFs from the QURE Benchmark agg suite.

Source: ``/home/michael/Downloads/QURE Benchmark/agg/q*.py``.

Each source file wraps ``udf_0`` as a PySpark ``pandas_udf`` of type
``GROUPED_AGG`` — it receives one or more ``pd.Series`` arguments (and
sometimes a scalar parameter) and returns a scalar (or, for a few UDFs,
a Series). The bodies are copied here verbatim into
``@hipy.compiled_function`` wrappers, dropping the pyspark decorators
and the unused ``import pyspark ...`` lines.

A few of the source UDFs reference undefined names (``aligned_data``,
``receipt_dates``, ``total_days`` used before assignment). Those are
unrunnable in stock pandas too; they are included as ``xfail`` tests so
the source set stays visible but the suite doesn't pretend they
"succeed".

Each test builds the input Series from a small lineitem-like /
orders-like fixture using ``not_constant`` and prints the UDF result.
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
    # Small lineitem fixture. Columns cover what the agg UDFs need —
    # shipinstruct / shipdate / commitdate in addition to the map-suite
    # fields.
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
        "l_shipinstruct": not_constant(["URGENT", "NONE", "URGENT"]),
        "l_shipdate": not_constant(["2024-01-10", "2024-02-10", "2024-03-10"]),
        "l_commitdate": not_constant(["2024-01-05", "2024-02-15", "2024-03-01"]),
    })


@hipy.compiled_function
def _orders_df():
    return pd.DataFrame({
        "o_orderkey": not_constant([1, 2, 3]),
        "o_custkey": not_constant([10, 10, 20]),
        "o_orderdate": not_constant(["2024-01-01", "2024-01-15", "2024-03-01"]),
        "o_totalprice": not_constant([100.0, 200.0, 300.0]),
        "o_shippriority": not_constant([1, 2, 5]),
    })


# ---------------------------------------------------------------------------
# q1 — mask + series-series product sum
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q1(l_extendedprice, l_tax):
    # Validate tax rates
    valid_taxes = l_tax[(l_tax >= 0) & (l_tax <= 1)]
    tax_amounts = l_extendedprice * valid_taxes
    total_tax = tax_amounts.sum()
    return total_tax


@hipy.compiled_function
def fn_q1():
    df = _lineitem_df()
    print(udf_q1(df["l_extendedprice"], df["l_tax"]))


def test_q1():
    # 1000*0.1 + 5000*0.2 + 15000*0.3 = 100 + 1000 + 4500 = 5600.
    check_prints(fn_q1, "5600.0")


# ---------------------------------------------------------------------------
# q2 — string delimiter.join over a Series
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q2(l_comment, delimiter):
    # Concatenate comments
    concatenated = delimiter.join(l_comment)
    return concatenated


@hipy.compiled_function
def fn_q2():
    df = _lineitem_df()
    print(udf_q2(df["l_comment"], ","))


def test_q2():
    check_prints(fn_q2, "fast,slow,on time")


# ---------------------------------------------------------------------------
# q3 — uses `total_days` before assignment, and undefined `receipt_dates`
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q3(l_orderkey, l_quantity, l_shipdate, l_commitdate):
    total_orders = l_orderkey.nunique()
    #total_days = (pd.to_datetime(l_shipdate, errors='coerce').max() - pd.to_datetime(l_shipdate, errors='coerce').min()).days + 1
    if total_days > 0:
        order_frequency = total_orders / total_days
    else:
        order_frequency = 0.0
    average_order_size = l_quantity.mean()
    ship_dates = to_datetime(l_shipdate, 'coerce')
    commit_dates = to_datetime(l_commitdate, 'coerce')
    valid_ship_days = ship_dates[ship_dates.notnull() & receipt_dates.notnull()]
    valid_commit_dates = commit_dates[ship_dates.notnull() & receipt_dates.notnull()]
    on_time = valid_ship_days <= valid_commit_dates
    on_time_deliveries = on_time.sum()
    total_deliveries = len(valid_ship_days)
    if total_deliveries > 0.0:
        timely_delivery_ratio = (on_time_deliveries / total_deliveries) * 100.0
    else:
        timely_delivery_ratio = 0.0
    engagement_score = (order_frequency * 0.3) + (average_order_size * 0.4) + (timely_delivery_ratio * 0.3)
    engagement_score = min(engagement_score, 100.0)
    return engagement_score


@hipy.compiled_function
def fn_q3():
    df = _lineitem_df()
    print(udf_q3(df["l_orderkey"], df["l_quantity"], df["l_shipdate"], df["l_commitdate"]))


@pytest.mark.xfail(reason="Source UDF uses `total_days` before assignment and `receipt_dates` "
                          "which is never defined — unrunnable in stock pandas too.")
def test_q3():
    check_prints(fn_q3, "")


# ---------------------------------------------------------------------------
# q4 — pd.to_datetime + PeriodIndex arithmetic + std/mean
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q4(o_orderdate, o_totalprice):
    # Convert order dates to datetime
    order_dates = pd.to_datetime(o_orderdate, errors='coerce')
    min_date = order_dates.min()
    max_date = order_dates.max()
    total_months = (max_date.to_period('M') - min_date.to_period('M')).n + 1
    total_orders = len(order_dates)
    if total_months > 0:
        average_order_frequency = total_orders / total_months
    else:
        average_order_frequency = 0.0
    average_order_size = o_totalprice.mean()
    std_dev_order_amount = o_totalprice.std()
    if average_order_size != 0:
        coeff_variation = (std_dev_order_amount / average_order_size) * 100
    else:
        coeff_variation = 0.0
    behavior_score = (average_order_frequency * 0.4) + (average_order_size * 0.3) - (coeff_variation * 0.3)
    behavior_score = max(behavior_score, 0)
    return behavior_score


@hipy.compiled_function
def fn_q4():
    df = _orders_df()
    print(udf_q4(df["o_orderdate"], df["o_totalprice"]))


def test_q4():
    check_prints(fn_q4, "45.4")


# ---------------------------------------------------------------------------
# q5 — Series.map(dict)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q5(l_shipmode):
    # Define shipping costs per mode
    shipping_costs = {
        'AIR': 50.0,
        'RAIL': 30.0,
        'SHIP': 20.0,
        'TRUCK': 25.0,
        'MAIL': 15.0,
        'FOB': 40.0,
        'REG AIR': 55.0,
    }
    # Map ship modes to costs
    costs = l_shipmode.map(shipping_costs)
    total_cost = costs.sum()
    return total_cost


@hipy.compiled_function
def fn_q5():
    df = _lineitem_df()
    print(udf_q5(df["l_shipmode"]))


def test_q5():
    # AIR=50 + MAIL=15 + TRUCK=25 = 90.
    check_prints(fn_q5, "90.0")


# ---------------------------------------------------------------------------
# q6 — simple net-profit / margin arithmetic
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q6(l_extendedprice, l_discount, l_tax):
    operational_costs_percentage = 0.10
    net_sales = (l_extendedprice * (1.0 - l_discount)).sum()
    total_tax = (l_extendedprice * l_tax).sum()
    gross_profit = net_sales - total_tax
    operational_costs = net_sales * operational_costs_percentage
    net_profit = gross_profit - operational_costs
    if (net_sales != 0.0):
        profit_margin = (net_profit / net_sales) * 100.0
    else:
        profit_margin = 0.0
    return profit_margin


@hipy.compiled_function
def fn_q6():
    df = _lineitem_df()
    print(udf_q6(df["l_extendedprice"], df["l_discount"], df["l_tax"]))


def test_q6():
    check_prints(fn_q6, "53.63636363636364")


# ---------------------------------------------------------------------------
# q7 — Series.groupby(Series).sum().mean()
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q7(l_quantity, l_orderkey):
    # Group quantities by order key
    order_totals = l_quantity.groupby(l_orderkey).sum()
    average_size = order_totals.mean()
    return average_size


@hipy.compiled_function
def fn_q7():
    df = _lineitem_df()
    print(udf_q7(df["l_quantity"], df["l_orderkey"]))


def test_q7():
    # orderkeys are distinct so groupby.sum is 10, 20, 30; mean = 20.
    check_prints(fn_q7, "20.0")


# ---------------------------------------------------------------------------
# q8 — Series.corr(Series)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q8(l_quantity, l_extendedprice):
    # Check for sufficient data
    correlation = l_quantity.corr(l_extendedprice)
    return correlation


@hipy.compiled_function
def fn_q8():
    df = _lineitem_df()
    print(udf_q8(df["l_quantity"], df["l_extendedprice"]))


def test_q8():
    check_prints(fn_q8, "0.970725343394151")


# ---------------------------------------------------------------------------
# q9 — per-row margin, groupby(orderkey).mean() → Series-valued
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q9(orderkey, extendedprice, discount, supplycost):
    # Calculate revenue after discount
    revenue = extendedprice * (1.0 - discount)
    # Calculate profit as revenue minus supply cost
    profit = revenue - supplycost
    # Calculate profit margin as profit divided by revenue
    margin = (profit / revenue) * 100.0
    avg_margin = margin.groupby(orderkey).mean()
    return avg_margin


@hipy.compiled_function
def fn_q9():
    df = _lineitem_df()
    # supplycost is normally from partsupp; use a fabricated Series aligned
    # with the lineitem rows for this test.
    supplycost = pd.Series(not_constant([50.0, 100.0, 200.0]))
    print(udf_q9(df["l_orderkey"], df["l_extendedprice"], df["l_discount"], supplycost))


def test_q9():
    check_prints(fn_q9, """l_orderkey
1    94.444444
2    97.500000
3    98.095238
Name: l_extendedprice, dtype: float64""")


# ---------------------------------------------------------------------------
# q10 — string equality filter + .count() / len(); scalar APD
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q10(o_orderdate, o_shippriority, l_returnflag, average_payment_delay):
    total_items = len(l_returnflag)
    returned = l_returnflag[l_returnflag == 'R']
    returned_items = returned.count()
    if total_items > 0.0:
        return_frequency = (returned_items / total_items) * 100.0
    else:
        return_frequency = 0.0
    high_priority = o_shippriority[o_shippriority < 3]
    high_priority_orders = high_priority.count()
    total_orders = len(o_shippriority)
    if total_orders > 0.0:
        high_priority_ratio = (high_priority_orders / total_orders) * 100.0
    else:
        high_priority_ratio = 0.0
    risk_score = (average_payment_delay * 0.5) + (return_frequency * 0.3) - (high_priority_ratio * 0.2)
    risk_score = max(risk_score, 0)
    return risk_score


@hipy.compiled_function
def fn_q10():
    df = _lineitem_df()
    orders = _orders_df()
    print(udf_q10(orders["o_orderdate"], orders["o_shippriority"], df["l_returnflag"], 2))


def test_q10():
    check_prints(fn_q10, "0")


# ---------------------------------------------------------------------------
# q11 — Series[cond].count() with scalar parameter
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q11(l_discount, discount_threshold):
    high_discounts = l_discount[l_discount > discount_threshold]
    count = high_discounts.count()
    return count


@hipy.compiled_function
def fn_q11():
    df = _lineitem_df()
    print(udf_q11(df["l_discount"], 0.10))


def test_q11():
    check_prints(fn_q11, "2")


# ---------------------------------------------------------------------------
# q12 — string equality filter + count / len ratio
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q12(l_returnflag):
    total_items = len(l_returnflag)
    if total_items == 0:
        return 0.0
    returned_items = l_returnflag[l_returnflag == 'R']
    returned_items_count = len(returned_items)
    ratio = returned_items_count / total_items
    return ratio


@hipy.compiled_function
def fn_q12():
    df = _lineitem_df()
    print(udf_q12(df["l_returnflag"]))


def test_q12():
    check_prints(fn_q12, "0.3333333333333333")


# ---------------------------------------------------------------------------
# q13 — string equality filter + count / len ratio (same as q12)
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q13(l_shipinstruct):
    total_shipments = len(l_shipinstruct)
    if total_shipments == 0.0:
        return 0.0
    urgent_shipments = l_shipinstruct[l_shipinstruct == 'URGENT']
    urgent_shipments_count = len(urgent_shipments)
    ratio = urgent_shipments_count / total_shipments
    return ratio


@hipy.compiled_function
def fn_q13():
    df = _lineitem_df()
    print(udf_q13(df["l_shipinstruct"]))


def test_q13():
    check_prints(fn_q13, "0.6666666666666666")


# ---------------------------------------------------------------------------
# q14 — pd.to_datetime + .notnull() mask + .dt.days + .mean() / .empty
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q14(l_shipdate, l_commitdate):
    # Convert dates to datetime
    ship_dates = pd.to_datetime(l_shipdate, errors='coerce')
    commit_dates = pd.to_datetime(l_commitdate, errors='coerce')

    # Calculate delays, ignoring invalid dates
    valid_dates = ship_dates.notnull() & commit_dates.notnull()
    delays = (ship_dates[valid_dates] - commit_dates[valid_dates]).dt.days

    if delays.empty:
        return 0.0
    average_delay = delays.mean()
    return average_delay


@hipy.compiled_function
def fn_q14():
    df = _lineitem_df()
    print(udf_q14(df["l_shipdate"], df["l_commitdate"]))


def test_q14():
    check_prints(fn_q14, "3.0")


# ---------------------------------------------------------------------------
# q15 — (shipdate - commitdate).dt.days grouped by orderkey
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q15(orderkey, shipdate, commitdate):
    # Calculate delay in days
    delay = (shipdate - commitdate).dt.days
    # Calculate average delay per order
    avg_delay = delay.groupby(orderkey).mean()
    return avg_delay


@hipy.compiled_function
def fn_q15():
    df = _lineitem_df()
    # q15 expects shipdate / commitdate to already be datetime-typed,
    # so convert here before passing.
    sd = pd.to_datetime(df["l_shipdate"])
    cd = pd.to_datetime(df["l_commitdate"])
    print(udf_q15(df["l_orderkey"], sd, cd))


def test_q15():
    check_prints(fn_q15, """l_orderkey
1    5.0
2   -5.0
3    9.0
Name: l_shipdate, dtype: float64""")


# ---------------------------------------------------------------------------
# q16 — Series[cond].max()
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q16(l_discount):
    # Validate discounts
    valid_discounts = l_discount[(l_discount >= 0) & (l_discount <= 1)]
    max_discount = valid_discounts.max()
    return max_discount


@hipy.compiled_function
def fn_q16():
    df = _lineitem_df()
    print(udf_q16(df["l_discount"]))


def test_q16():
    check_prints(fn_q16, "0.3")


# ---------------------------------------------------------------------------
# q17 — inventory turnover, simple series arithmetic and .mean()
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q17(l_extendedprice, l_quantity, p_retailprice):
    cogs = l_extendedprice.sum()
    average_inventory_value = (p_retailprice * l_quantity).mean()
    if average_inventory_value == 0.0:
        return 0.0
    inventory_turnover = cogs / average_inventory_value
    return inventory_turnover


@hipy.compiled_function
def fn_q17():
    df = _lineitem_df()
    p_retail = pd.Series(not_constant([50.0, 100.0, 150.0]))
    print(udf_q17(df["l_extendedprice"], df["l_quantity"], p_retail))


def test_q17():
    check_prints(fn_q17, "9.0")


# ---------------------------------------------------------------------------
# q18 — uses undefined `aligned_data`
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q18(l_extendedprice, l_discount, l_tax):
    valid_prices = l_extendedprice[l_extendedprice >= 0]
    if valid_prices.empty:
        return 0.0
    valid_discounts = l_discount[(l_discount >= 0) & (l_discount <= 1)]
    valid_taxes = l_tax[(l_tax >= 0) & (l_tax <= 1)]
    discounted_price = aligned_data['price'] * (1 - aligned_data['discount'])
    aligned_data['discounted_price'] = discounted_price
    taxed_price = discounted_price * (1 + aligned_data['tax'])
    aligned_data['taxed_price'] = taxed_price
    total_revenue = taxed_price.sum()
    average_discounted_price = discounted_price.mean()
    total_tax_amount = (discounted_price * aligned_data['tax']).sum()
    return total_revenue


@hipy.compiled_function
def fn_q18():
    df = _lineitem_df()
    print(udf_q18(df["l_extendedprice"], df["l_discount"], df["l_tax"]))


@pytest.mark.xfail(reason="Source UDF uses undefined name `aligned_data` — unrunnable in "
                          "stock pandas too.")
def test_q18():
    check_prints(fn_q18, "")


# ---------------------------------------------------------------------------
# q19 — uses undefined `receipt_dates`
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q19(l_shipdate, l_commitdate):
    ship_dates = to_datetime(l_shipdate, 'coerce')
    commit_dates = to_datetime(l_commitdate, 'coerce')
    valid_ship_days = ship_dates[ship_dates.notnull() & receipt_dates.notnull()]
    valid_commit_days = commit_dates[ship_dates.notnull() & receipt_dates.notnull()]
    delays = valid_ship_days - valid_commit_days
    delays_days = delays.dt.days
    delays_non_negative = delays_days[delays_days >= 0]  # Consider only non-negative delays
    if len(delays_non_negative) > 0.0:
        average_delay = delays_non_negative.mean()
        max_delay = delays_non_negative.max()
    else:
        average_delay = 0.0
        max_delay = 0.0
    late_shipments_count = len(delays_non_negative)
    total_shipments = len(delays)
    if total_shipments > 0.0:
        late_shipments_percentage = (late_shipments_count / total_shipments) * 100.0
    else:
        late_shipments_percentage = 0.0
    performance_score = 100.0 - (average_delay * 2.0 + late_shipments_percentage)
    performance_score = max(performance_score, 0.0)
    return performance_score


@hipy.compiled_function
def fn_q19():
    df = _lineitem_df()
    print(udf_q19(df["l_shipdate"], df["l_commitdate"]))


@pytest.mark.xfail(reason="Source UDF uses undefined `receipt_dates` — unrunnable in stock "
                          "pandas too.")
def test_q19():
    check_prints(fn_q19, "")


# ---------------------------------------------------------------------------
# q20 — CLV with datetime min/max and day diff
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q20(o_orderdate, o_totalprice):
    order_dates = pd.to_datetime(o_orderdate, errors='coerce')
    if order_dates.isnull().all():
        return 0.0
    average_purchase_value = o_totalprice.mean()
    total_orders = len(order_dates)
    min_date = order_dates.min()
    max_date = order_dates.max()
    total_days = (max_date - min_date).days + 1
    if total_days > 0:
        purchase_frequency = total_orders / total_days
    else:
        purchase_frequency = 0.0
    average_lifespan_days = 365  # One year
    clv = average_purchase_value * purchase_frequency * average_lifespan_days
    return clv


@hipy.compiled_function
def fn_q20():
    df = _orders_df()
    print(udf_q20(df["o_orderdate"], df["o_totalprice"]))


def test_q20():
    check_prints(fn_q20, "3590.1639344262294")


# ---------------------------------------------------------------------------
# q21 — groupby + nunique / mean product → Series-valued
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q21(custkey, orderkey, totalprice):
    orders_count = orderkey.groupby(custkey).nunique()
    avg_order_size = totalprice.groupby(custkey).mean()
    loyalty_index = orders_count * avg_order_size
    return loyalty_index


@hipy.compiled_function
def fn_q21():
    df = _orders_df()
    print(udf_q21(df["o_custkey"], df["o_orderkey"], df["o_totalprice"]))


def test_q21():
    check_prints(fn_q21, """o_custkey
10    300.0
20    300.0
dtype: float64""")


# ---------------------------------------------------------------------------
# q22 — Series.var()
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q22(l_quantity):
    if len(l_quantity) < 2:
        return 0.0
    variance = l_quantity.var()
    return variance


@hipy.compiled_function
def fn_q22():
    df = _lineitem_df()
    print(udf_q22(df["l_quantity"]))


def test_q22():
    check_prints(fn_q22, "100.0")


# ---------------------------------------------------------------------------
# q23 — full-range discount / tax validity check + total revenue
# ---------------------------------------------------------------------------
@hipy.compiled_function
def udf_q23(l_extendedprice, l_discount, l_tax):
    valid_discounts = l_discount[(l_discount >= 0) & (l_discount <= 1)]
    valid_taxes = l_tax[(l_tax >= 0) & (l_tax <= 1)]

    if len(valid_discounts) != len(l_discount) or len(valid_taxes) != len(l_tax):
        return 0.0

    discounted_price = l_extendedprice * (1.0 - l_discount)
    taxed_price = discounted_price * (1.0 + l_tax)
    total_revenue = taxed_price.sum()

    # Return the total revenue
    return total_revenue


@hipy.compiled_function
def fn_q23():
    df = _lineitem_df()
    print(udf_q23(df["l_extendedprice"], df["l_discount"], df["l_tax"]))


def test_q23():
    check_prints(fn_q23, "19440.0")
