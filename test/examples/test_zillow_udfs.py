"""Tests for UDFs from the zillow-hipy benchmark
(`udf-benchmark/benchmarks/lingodb/zillow-hipy/initialize.sql`).
"""
import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.math
import math
import hipy.lib.re
import re


@hipy.compiled_function
def extractbd(val):
    try:
        max_idx = val.find(' bd')
        if max_idx < 0:
            max_idx = len(val)
        s = val[:max_idx]
        split_idx = s.rfind(',')
        if split_idx < 0:
            split_idx = 0
        else:
            split_idx += 2
        r = s[split_idx:]
        return int(r)
    except:
        return -1


@hipy.compiled_function
def fn_extractbd():
    print(extractbd(not_constant("3 bd, 2 ba, 1500 sqft")))
    print(extractbd(not_constant("no bd here")))


def test_extractbd():
    check_prints(fn_extractbd, """
3
-1
""")


@hipy.compiled_function
def extractba(val):
    try:
        max_idx = val.find(' ba')
        if max_idx < 0:
            max_idx = len(val)
        s = val[:max_idx]
        split_idx = s.rfind(',')
        if split_idx < 0:
            split_idx = 0
        else:
            split_idx += 2
        r = s[split_idx:]
        ba = math.ceil(2.0 * float(r)) / 2.0
        return int(ba)
    except:
        return -1


@hipy.compiled_function
def fn_extractba():
    print(extractba(not_constant("3 bd, 2.5 ba, 1500 sqft")))
    print(extractba(not_constant("no ba")))


def test_extractba():
    check_prints(fn_extractba, """
2
-1
""")


@hipy.compiled_function
def extractsqfeet(val):
    try:
        max_idx = val.find(' sqft')
        if max_idx < 0:
            max_idx = len(val)
        s = val[:max_idx]
        split_idx = s.rfind('ba ,')
        if split_idx < 0:
            split_idx = 0
        else:
            split_idx += 5
        r = s[split_idx:]
        r = r.replace(',', '')
        return int(r)
    except:
        return -1


@hipy.compiled_function
def fn_extractsqfeet():
    print(extractsqfeet(not_constant("3 bd, 2 ba , 1,500 sqft")))
    print(extractsqfeet(not_constant("no sqft")))


def test_extractsqfeet():
    check_prints(fn_extractsqfeet, """
1500
-1
""")


@hipy.compiled_function
def extractid(val):
    match = re.search("(\d+)_zpid/$", val)
    try:
        return int(match.group(1))
    except:
        return 0


@hipy.compiled_function
def fn_extractid():
    print(extractid(not_constant("http://example.com/123456_zpid/")))
    print(extractid(not_constant("no id")))


def test_extractid():
    check_prints(fn_extractid, """
123456
0
""")


@hipy.compiled_function
def extractprice_sell(val):
    try:
        return int(val[1:].replace(',', ''))
    except:
        return -1


@hipy.compiled_function
def fn_extractprice_sell():
    print(extractprice_sell(not_constant("$1,500,000")))
    print(extractprice_sell(not_constant("")))


def test_extractprice_sell():
    check_prints(fn_extractprice_sell, """
1500000
-1
""")


@hipy.compiled_function
def extracttype(val):
    try:
        t = val.lower()
        type = 'unknown'
        if 'condo' in t or 'apartment' in t:
            type = 'condo'
        if 'house' in t:
            type = 'house'
        return type
    except:
        return 'null'


@hipy.compiled_function
def fn_extracttype():
    print(extracttype(not_constant("House in nice area")))
    print(extracttype(not_constant("Condo on top floor")))
    print(extracttype(not_constant("Apartment downtown")))
    print(extracttype(not_constant("Studio lot")))


def test_extracttype():
    check_prints(fn_extracttype, """
house
condo
condo
unknown
""")


@hipy.compiled_function
def extractpcode(val):
    try:
        return '%05d' % int(val)
    except:
        return ''


@hipy.compiled_function
def fn_extractpcode():
    print(extractpcode(not_constant("98101")))
    print(extractpcode(not_constant("42")))
    print("[" + extractpcode(not_constant("xx")) + "]")


def test_extractpcode():
    check_prints(fn_extractpcode, """
98101
00042
[]
""")
