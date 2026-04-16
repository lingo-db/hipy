import hipy
import hipy.lib.datetime
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import datetime
from datetime import date as py_date


@hipy.compiled_function
def fn_construct():
    d = datetime.date(2024, 5, 16)
    print(d)
    print(d.year)
    print(d.month)
    print(d.day)
    y = not_constant(2024)
    m = not_constant(5)
    day = not_constant(16)
    d2 = datetime.date(y, m, day)
    print(d2)
    print(d2.year)
    print(d2.month)
    print(d2.day)


def test_construct():
    check_prints(fn_construct, """
2024-05-16
2024
5
16
2024-05-16
2024
5
16""")


@hipy.compiled_function
def fn_fromisoformat():
    d = datetime.date.fromisoformat("2024-05-16")
    print(d)
    print(d.year, d.month, d.day)
    s = not_constant("2023-01-02")
    d2 = datetime.date.fromisoformat(s)
    print(d2)


def test_fromisoformat():
    check_prints(fn_fromisoformat, """
2024-05-16
2024 5 16
2023-01-02""")


@hipy.compiled_function
def fn_isoformat():
    d = datetime.date(2024, 5, 16)
    print(d.isoformat())


def test_isoformat():
    check_prints(fn_isoformat, """
2024-05-16""")


@hipy.compiled_function
def fn_date_compare():
    a = datetime.date(2024, 5, 16)
    b = datetime.date(2024, 5, 17)
    print(a == a)
    print(a == b)
    print(a != b)
    print(a < b)
    print(b > a)
    print(a <= a)
    print(a >= a)
    print(a < a)
    print(a > a)


def test_date_compare():
    check_prints(fn_date_compare, """
True
False
True
True
True
True
True
False
False""")


@hipy.compiled_function
def fn_date_diff():
    a = datetime.date(2024, 5, 20)
    b = datetime.date(2024, 5, 16)
    diff = a - b
    print(diff.days)
    print(diff.seconds)


def test_date_diff():
    check_prints(fn_date_diff, """
4
0""")


@hipy.compiled_function
def fn_date_add_timedelta():
    a = datetime.date(2024, 5, 16)
    delta = datetime.timedelta(7)
    b = a + delta
    print(b)
    c = a - delta
    print(c)
    d = delta + a
    print(d)


def test_date_add_timedelta():
    check_prints(fn_date_add_timedelta, """
2024-05-23
2024-05-09
2024-05-23""")


@hipy.compiled_function
def fn_weekday():
    # 2024-05-16 is a Thursday -> weekday == 3, isoweekday == 4
    d = datetime.date(2024, 5, 16)
    print(d.weekday())
    print(d.isoweekday())


def test_weekday():
    check_prints(fn_weekday, """
3
4""")


@hipy.compiled_function
def fn_toordinal():
    # Ordinal ordering: days between two dates should match their .toordinal() difference.
    a = datetime.date(2024, 5, 16)
    b = datetime.date(2024, 5, 20)
    print(b.toordinal() - a.toordinal())
    print(a.toordinal())


def test_toordinal():
    check_prints(fn_toordinal, f"""
4
{py_date(2024, 5, 16).toordinal()}""")


@hipy.compiled_function
def fn_timedelta_construct():
    t = datetime.timedelta(5)
    print(t.days)
    print(t.seconds)
    t2 = datetime.timedelta(5, 3600)
    print(t2.days)
    print(t2.seconds)
    d = not_constant(3)
    s = not_constant(120)
    t3 = datetime.timedelta(d, s)
    print(t3.days)
    print(t3.seconds)


def test_timedelta_construct():
    check_prints(fn_timedelta_construct, """
5
0
5
3600
3
120""")


@hipy.compiled_function
def fn_timedelta_arith():
    a = datetime.timedelta(5)
    b = datetime.timedelta(3)
    print((a + b).days)
    print((a - b).days)
    print((-a).days)


def test_timedelta_arith():
    check_prints(fn_timedelta_arith, """
8
2
-5""")


@hipy.compiled_function
def fn_timedelta_compare():
    a = datetime.timedelta(5)
    b = datetime.timedelta(3)
    print(a == a)
    print(a == b)
    print(a != b)
    print(b < a)
    print(a > b)
    print(a <= a)
    print(a >= a)


def test_timedelta_compare():
    check_prints(fn_timedelta_compare, """
True
False
True
True
True
True
True""")


@hipy.compiled_function
def fn_timedelta_total_seconds():
    t = datetime.timedelta(1, 30)
    print(t.total_seconds())


def test_timedelta_total_seconds():
    check_prints(fn_timedelta_total_seconds, """
86430.0""")


@hipy.compiled_function
def fn_timedelta_str():
    a = datetime.timedelta(0, 60)
    print(a)
    b = datetime.timedelta(1, 3661)
    print(b)
    c = datetime.timedelta(5, 0)
    print(c)


def test_timedelta_str():
    check_prints(fn_timedelta_str, """
0:01:00
1 day, 1:01:01
5 days, 0:00:00""")


@hipy.compiled_function
def fn_timedelta_bool():
    z = datetime.timedelta(0)
    if z:
        print("nonzero")
    else:
        print("zero")
    t = datetime.timedelta(1)
    if t:
        print("nonzero")
    else:
        print("zero")


def test_timedelta_bool():
    check_prints(fn_timedelta_bool, """
zero
nonzero""")


@hipy.compiled_function
def fn_date_through_variable():
    y = not_constant(2024)
    m = not_constant(12)
    d = not_constant(31)
    date1 = datetime.date(y, m, d)
    date2 = datetime.date(2025, 1, 1)
    diff = date2 - date1
    print(diff.days)


def test_date_through_variable():
    check_prints(fn_date_through_variable, """
1""")


