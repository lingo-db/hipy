"""Tests for UDFs from the procbenchpy-hipy benchmark
(`udf-benchmark/benchmarks/lingodb/procbenchpy-hipy/initialize.sql`).

Every UDF in this benchmark calls `hipy.lib.sql.execute(...)` against
TPC-DS tables. Per `docs/standard-library.md` the `sql.execute` builtin
is only lowered by the MLIR backend — the C++ backend (which
`check_prints` drives) does not implement it.

The wrappers below are written so that the UDF bodies are verbatim from
the SQL file and exercised with `not_constant(...)` inputs, but each
`test_` is marked skipped because running them under the current
backend fails with `NotImplementedError: builtin sql.execute not
implemented`. Remove the skip once the MLIR backend (or another
execution harness for these UDFs) is wired into the test runner.
"""
import pytest

import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.sql as sql


@hipy.compiled_function
def increase_in_web_spending(cust_sk):
    spending1 = 0
    spending2 = 0
    increase = 0

    r = sql.execute(
        sql.nullable(float),
        "SELECT SUM(ws_net_paid_inc_ship_tax)::float "
        "FROM web_sales, date_dim "
        "WHERE d_date_sk = ws_sold_date_sk "
        "  AND d_year = 2001 "
        "  AND ws_bill_customer_sk = PARAM(1)",
        cust_sk
    )
    spending1 = r.get_value_or_default(0.0)

    r = sql.execute(
        sql.nullable(float),
        "SELECT SUM(ws_net_paid_inc_ship_tax)::float "
        "FROM web_sales, date_dim "
        "WHERE d_date_sk = ws_sold_date_sk "
        "  AND d_year = 2000 "
        "  AND ws_bill_customer_sk = PARAM(1)",
        cust_sk
    )
    spending2 = r.get_value_or_default(0.0)

    if spending1 < spending2:
        return -1.0
    else:
        increase = spending1 - spending2
        return increase


@hipy.compiled_function
def fn_increase_in_web_spending():
    print(increase_in_web_spending(not_constant(1)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_increase_in_web_spending():
    check_prints(fn_increase_in_web_spending, "")


@hipy.compiled_function
def max_purchase_channel(ckey, from_date_sk, to_date_sk):
    num_sales_from_store = 0
    num_sales_from_catalog = 0
    num_sales_from_web = 0
    max_channel = ''

    num_sales_from_store = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM store_sales "
        "WHERE ss_customer_sk = PARAM(1) "
        "  AND ss_sold_date_sk >= PARAM(2) "
        "  AND ss_sold_date_sk <= PARAM(3)",
        ckey, from_date_sk, to_date_sk
    ).get_value_or_default(0)

    num_sales_from_catalog = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM catalog_sales "
        "WHERE cs_bill_customer_sk = PARAM(1) "
        "  AND cs_sold_date_sk >= PARAM(2) "
        "  AND cs_sold_date_sk <= PARAM(3)",
        ckey, from_date_sk, to_date_sk
    ).get_value_or_default(0)

    num_sales_from_web = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM web_sales "
        "WHERE ws_bill_customer_sk = PARAM(1) "
        "  AND ws_sold_date_sk >= PARAM(2) "
        "  AND ws_sold_date_sk <= PARAM(3)",
        ckey, from_date_sk, to_date_sk
    ).get_value_or_default(0)

    if num_sales_from_store > num_sales_from_catalog:
        max_channel = 'Store'
        if num_sales_from_web > num_sales_from_store:
            max_channel = 'Web'
    else:
        max_channel = 'Catalog'
        if num_sales_from_web > num_sales_from_catalog:
            max_channel = 'Web'

    return max_channel


@hipy.compiled_function
def fn_max_purchase_channel():
    print(max_purchase_channel(not_constant(1), not_constant(2000), not_constant(2001)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_max_purchase_channel():
    check_prints(fn_max_purchase_channel, "")


@hipy.compiled_function
def income_band_of_max_buy_customer(store_number):
    incomeband = 0
    cust = 0
    hhdemo = 0
    cnt_var = 0
    c_level = ''

    cust = sql.execute(
        sql.nullable(int),
        "SELECT ss_customer_sk::bigint "
        "FROM store_sales, customer "
        "WHERE ss_store_sk = Param(1) AND c_customer_sk = ss_customer_sk "
        "GROUP BY ss_customer_sk, c_current_hdemo_sk "
        "HAVING COUNT(*) = ("
        "  SELECT MAX(cnt) FROM ("
        "    SELECT ss_customer_sk, c_current_hdemo_sk, COUNT(*) as cnt "
        "    FROM store_sales, customer "
        "    WHERE ss_store_sk = PARAM(1) "
        "      AND c_customer_sk = ss_customer_sk "
        "    GROUP BY ss_customer_sk, c_current_hdemo_sk "
        "    HAVING ss_customer_sk IS NOT NULL"
        "  ) tbl"
        ") "
        "LIMIT 1", store_number
    ).get_value_or_default(-1)

    hhdemo = sql.execute(
        sql.nullable(int),
        "SELECT c_current_hdemo_sk::bigint "
        "FROM store_sales, customer "
        "WHERE ss_store_sk = Param(1) AND c_customer_sk = ss_customer_sk "
        "GROUP BY ss_customer_sk, c_current_hdemo_sk "
        "HAVING COUNT(*) = ("
        "  SELECT MAX(cnt) FROM ("
        "    SELECT ss_customer_sk, c_current_hdemo_sk, COUNT(*) as cnt "
        "    FROM store_sales, customer "
        "    WHERE ss_store_sk = PARAM(1) "
        "      AND c_customer_sk = ss_customer_sk "
        "    GROUP BY ss_customer_sk, c_current_hdemo_sk "
        "    HAVING ss_customer_sk IS NOT NULL"
        "  ) tbl"
        ") "
        "LIMIT 1", store_number
    ).get_value_or_default(-1)

    incomeband = sql.execute(
        sql.nullable(int),
        "SELECT hd_income_band_sk::bigint "
        "FROM household_demographics "
        "WHERE hd_demo_sk = PARAM(1)",
        hhdemo
    ).get_value_or_default(-1)

    if incomeband >= 0 and incomeband <= 3:
        c_level = 'low'
    if incomeband >= 4 and incomeband <= 7:
        c_level = 'lowerMiddle'
    if incomeband >= 8 and incomeband <= 11:
        c_level = 'upperMiddle'
    if incomeband >= 12 and incomeband <= 16:
        c_level = 'high'
    if incomeband >= 17 and incomeband <= 20:
        c_level = 'affluent'

    return c_level


@hipy.compiled_function
def fn_income_band_of_max_buy_customer():
    print(income_band_of_max_buy_customer(not_constant(1)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_income_band_of_max_buy_customer():
    check_prints(fn_income_band_of_max_buy_customer, "")


@hipy.compiled_function
def preferred_channel_wrt_count(cust_key):
    num_web = 0
    num_store = 0
    num_cat = 0

    num_web = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM web_sales "
        "WHERE ws_bill_customer_sk = PARAM(1)",
        cust_key
    ).get_value_or_default(0)

    num_store = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM store_sales "
        "WHERE ss_customer_sk = PARAM(1)",
        cust_key
    ).get_value_or_default(0)

    num_cat = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM catalog_sales "
        "WHERE cs_bill_customer_sk = PARAM(1)",
        cust_key
    ).get_value_or_default(0)

    if num_web >= num_store and num_web >= num_cat:
        return 'web'
    if num_store >= num_web and num_store >= num_cat:
        return 'store'
    if num_cat >= num_store and num_cat >= num_web:
        return 'Catalog'

    return 'Logical error'


@hipy.compiled_function
def fn_preferred_channel_wrt_count():
    print(preferred_channel_wrt_count(not_constant(1)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_preferred_channel_wrt_count():
    check_prints(fn_preferred_channel_wrt_count, "")


@hipy.compiled_function
def preferred_channel_wrt_expenditure(cust_key):
    num_web = 0.0
    num_store = 0.0
    num_cat = 0.0

    count_web = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM web_sales "
        "WHERE ws_bill_customer_sk = PARAM(1)",
        cust_key
    ).get_value_or_default(0)
    if count_web > 0:
        r = sql.execute(
            sql.nullable(float),
            "SELECT SUM(ws_net_paid_inc_ship_tax)::float "
            "FROM web_sales "
            "WHERE ws_bill_customer_sk = PARAM(1)",
            cust_key
        )
        num_web = r.get_value_or_default(0.0)

    count_store = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM store_sales "
        "WHERE ss_customer_sk = PARAM(1)",
        cust_key
    ).get_value_or_default(0)
    if count_store > 0:
        r = sql.execute(
            sql.nullable(float),
            "SELECT SUM(ss_net_paid_inc_tax)::float "
            "FROM store_sales "
            "WHERE ss_customer_sk = PARAM(1)",
            cust_key
        )
        num_store = r.get_value_or_default(0.0)

    count_cat = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM catalog_sales "
        "WHERE cs_bill_customer_sk = PARAM(1)",
        cust_key
    ).get_value_or_default(0)
    if count_cat > 0:
        r = sql.execute(
            sql.nullable(float),
            "SELECT SUM(cs_net_paid_inc_ship_tax)::float "
            "FROM catalog_sales "
            "WHERE cs_bill_customer_sk = PARAM(1)",
            cust_key
        )
        num_cat = r.get_value_or_default(0.0)

    if num_web >= num_store and num_web >= num_cat:
        return 'web'
    if num_store >= num_web and num_store >= num_cat:
        return 'store'
    if num_cat >= num_store and num_cat >= num_web:
        return 'Catalog'

    return 'Logical error'


@hipy.compiled_function
def fn_preferred_channel_wrt_expenditure():
    print(preferred_channel_wrt_expenditure(not_constant(1)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_preferred_channel_wrt_expenditure():
    check_prints(fn_preferred_channel_wrt_expenditure, "")


@hipy.compiled_function
def total_large_purchases(given_state, amount, yr, qtr):
    large_purchase = 0

    r = sql.execute(
        sql.nullable(float),
        "SELECT SUM(cs_net_paid_inc_ship_tax)::float "
        "FROM catalog_sales, customer, customer_address, date_dim "
        "WHERE cs_bill_customer_sk = c_customer_sk "
        "  AND c_current_addr_sk = ca_address_sk "
        "  AND ca_state = PARAM(1) "
        "  AND cs_net_paid_inc_ship_tax >= PARAM(2) "
        "  AND d_date_sk = cs_sold_date_sk "
        "  AND d_year = PARAM(3) "
        "  AND d_qoy = PARAM(4)",
        given_state, amount, yr, qtr
    )
    large_purchase = r.get_value_or_default(0.0)

    return large_purchase


@hipy.compiled_function
def fn_total_large_purchases():
    print(total_large_purchases(not_constant("CA"), not_constant(1000.0), not_constant(2001), not_constant(1)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_total_large_purchases():
    check_prints(fn_total_large_purchases, "")


@hipy.compiled_function
def get_manufact_complex(itm):
    man = ''
    cnt1 = 0
    cnt2 = 0

    cnt1 = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM store_sales, date_dim "
        "WHERE ss_item_sk = PARAM(1) "
        "  AND d_date_sk = ss_sold_date_sk "
        "  AND d_year = 2003",
        itm
    ).get_value_or_default(0)

    cnt2 = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM catalog_sales, date_dim "
        "WHERE cs_item_sk = PARAM(1) "
        "  AND d_date_sk = cs_sold_date_sk "
        "  AND d_year = 2003",
        itm
    ).get_value_or_default(0)

    if cnt1 > 0 and cnt2 > 0:
        man = sql.execute(
            sql.nullable(str),
            "SELECT i_manufact "
            "FROM item "
            "WHERE i_item_sk = PARAM(1)",
            itm
        ).get_value_or_default('')
    else:
        man = 'outdated item'

    return man


@hipy.compiled_function
def fn_get_manufact_complex():
    print(get_manufact_complex(not_constant(1)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_get_manufact_complex():
    check_prints(fn_get_manufact_complex, "")


@hipy.compiled_function
def morning_to_eve_ratio(dep):
    morning_sale = 0
    evening_sale = 0
    ratio = 0.0

    morning_sale = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM web_sales, time_dim, customer_demographics "
        "WHERE ws_sold_time_sk = t_time_sk "
        "  AND ws_bill_customer_sk = cd_demo_sk "
        "  AND t_hour >= 8 AND t_hour <= 9 "
        "  AND cd_dep_count = PARAM(1)",
        dep
    ).get_value_or_default(0)

    evening_sale = sql.execute(
        sql.nullable(int),
        "SELECT COUNT(*) "
        "FROM web_sales, time_dim, customer_demographics "
        "WHERE ws_sold_time_sk = t_time_sk "
        "  AND ws_bill_customer_sk = cd_demo_sk "
        "  AND t_hour >= 19 AND t_hour <= 20 "
        "  AND cd_dep_count = PARAM(1)",
        dep
    ).get_value_or_default(0)

    ratio = float(morning_sale) / float(evening_sale)
    return ratio


@hipy.compiled_function
def fn_morning_to_eve_ratio():
    print(morning_to_eve_ratio(not_constant(1)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_morning_to_eve_ratio():
    check_prints(fn_morning_to_eve_ratio, "")


@hipy.compiled_function
def total_discount(manufacture_id):
    average = 0.0
    addition = 0.0

    r = sql.execute(
        sql.nullable(float),
        "SELECT AVG(ws_ext_discount_amt)::float "
        "FROM web_sales, item "
        "WHERE ws_item_sk = i_item_sk "
        "  AND i_manufact_id = PARAM(1)",
        manufacture_id
    )
    average = r.get_value_or_default(0.0)

    r = sql.execute(
        sql.nullable(float),
        "SELECT SUM(ws_ext_discount_amt)::float "
        "FROM web_sales, item "
        "WHERE ws_item_sk = i_item_sk "
        "  AND i_manufact_id = PARAM(1) "
        "  AND ws_ext_discount_amt > 1.3 * PARAM(2)",
        manufacture_id, average
    )
    addition = r.get_value_or_default(0.0)

    return addition


@hipy.compiled_function
def fn_total_discount():
    print(total_discount(not_constant(1)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_total_discount():
    check_prints(fn_total_discount, "")


@hipy.compiled_function
def profitable_manager(manager, yr):
    net_profit = 0.0

    r = sql.execute(
        sql.nullable(float),
        "SELECT SUM(ss_net_profit)::float "
        "FROM store, store_sales, date_dim "
        "WHERE ss_sold_date_sk = d_date_sk "
        "  AND d_year = PARAM(2) "
        "  AND s_manager = PARAM(1) "
        "  AND s_store_sk = ss_store_sk",
        manager, yr
    )
    net_profit = r.get_value_or_default(0.0)

    if net_profit > 0:
        return 1
    else:
        return 0


@hipy.compiled_function
def fn_profitable_manager():
    print(profitable_manager(not_constant("Jane Doe"), not_constant(2001)))


@pytest.mark.skip(reason="sql.execute is not implemented by the C++ backend")
def test_profitable_manager():
    check_prints(fn_profitable_manager, "")
