"""Tests for UDFs from the flights-hipy benchmark
(`udf-benchmark/benchmarks/lingodb/flights-hipy/initialize.sql`).

Each UDF is copy-pasted verbatim from the SQL file, wrapped in a
compiled caller that feeds `not_constant(...)` inputs and prints the
result.
"""
import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def getcity(x):
    return x[:x.rfind(',')]


@hipy.compiled_function
def fn_getcity():
    print(getcity(not_constant("Seattle, WA")))
    print(getcity(not_constant("New York, NY")))


def test_getcity():
    check_prints(fn_getcity, """
Seattle
New York
""")


@hipy.compiled_function
def getstate(x):
    return x[x.rfind(',')+1:]


@hipy.compiled_function
def fn_getstate():
    print(getstate(not_constant("Seattle, WA")))
    print(getstate(not_constant("New York, NY")))


def test_getstate():
    check_prints(fn_getstate, """
 WA
 NY
""")


@hipy.compiled_function
def toint(val):
    return int(float(val)) if val else 0


@hipy.compiled_function
def fn_toint():
    print(toint(not_constant("42")))
    print(toint(not_constant("3.7")))
    print(toint(not_constant("")))


def test_toint():
    check_prints(fn_toint, """
42
3
0
""")


@hipy.compiled_function
def getairlineyear(arg):
    try:
        return int(arg[arg.rfind('(') + 1:arg.rfind('-')])
    except:
        return -1


@hipy.compiled_function
def fn_getairlineyear():
    print(getairlineyear(not_constant("American Airlines (1960-2001)")))
    print(getairlineyear(not_constant("Virgin America (2007-)")))
    print(getairlineyear(not_constant("Bad Name")))


def test_getairlineyear():
    check_prints(fn_getairlineyear, """
1960
2007
-1
""")


@hipy.compiled_function
def getairlinename(inp):
    inp = inp[:inp.rfind('(')].strip()
    return inp.replace('Inc.', '').replace('LLC', '').replace('Co.', '').strip()


@hipy.compiled_function
def fn_getairlinename():
    print(getairlinename(not_constant("American Airlines Inc. (1960-2001)")))
    print(getairlinename(not_constant("United Airlines LLC (1926-)")))
    print(getairlinename(not_constant("Southwest Co. (1967-)")))


def test_getairlinename():
    check_prints(fn_getairlinename, """
American Airlines
United Airlines
Southwest
""")


@hipy.compiled_function
def defunctyear(arg):
    desc = arg[arg.rfind('-') + 1:arg.rfind(')')].strip()
    try:
        return int(desc) if len(desc) > 0 else -1
    except:
        return -1


@hipy.compiled_function
def fn_defunctyear():
    print(defunctyear(not_constant("Pan Am (1927-1991)")))
    print(defunctyear(not_constant("Virgin America (2007-)")))
    print(defunctyear(not_constant("Malformed")))


def test_defunctyear():
    check_prints(fn_defunctyear, """
1991
-1
-1
""")


@hipy.compiled_function
def gettime(val):
    return '{:02}:{:02}'.format(int(val / 100), val % 100) if val else ''


@hipy.compiled_function
def fn_gettime():
    # gettime passes a float into `'{:02}'.format(...)`. Python would emit
    # `45.0`; HiPy's format-spec translation (`translate_python_spec_to_cpp`)
    # forwards `{:02}` to C++20 std::format, which prints a double as its
    # shortest round-trip form — so `45.0` comes out as `45`.
    print("[" + gettime(not_constant(0.0)) + "]")
    print(gettime(not_constant(1245.0)))
    print(gettime(not_constant(730.0)))


def test_gettime():
    check_prints(fn_gettime, """
[]
12:45
07:30
""")


@hipy.compiled_function
def cancelledbool(arg):
    return bool(arg)


@hipy.compiled_function
def fn_cancelledbool():
    print(cancelledbool(not_constant(1.0)))
    print(cancelledbool(not_constant(0.0)))
    print(cancelledbool(not_constant(2.5)))


def test_cancelledbool():
    check_prints(fn_cancelledbool, """
True
False
True
""")


@hipy.compiled_function
def _cancelledbool_dm(arg):
    return bool(arg)


@hipy.compiled_function
def cleanCode(code):
    if code == 'A':
        return 'carrier'
    elif code == 'B':
        return 'weather'
    elif code == 'C':
        return 'national air system'
    elif code == 'D':
        return 'security'
    else:
        return ''


@hipy.compiled_function
def divertedmap(arg1, arg2):
    if _cancelledbool_dm(arg1):
        return 'diverted'
    else:
        ccode = cleanCode(arg2)
        if ccode != '':
            return ccode
        else:
            return 'None'


@hipy.compiled_function
def fn_divertedmap():
    print(divertedmap(not_constant(1.0), not_constant("")))
    print(divertedmap(not_constant(0.0), not_constant("A")))
    print(divertedmap(not_constant(0.0), not_constant("B")))
    print(divertedmap(not_constant(0.0), not_constant("C")))
    print(divertedmap(not_constant(0.0), not_constant("D")))
    print(divertedmap(not_constant(0.0), not_constant("X")))


def test_divertedmap():
    check_prints(fn_divertedmap, """
diverted
carrier
weather
national air system
security
None
""")


@hipy.compiled_function
def fillintimes(c, d, e):
    if d != "":
        if float(d) > 0:
            return float(e)
        else:
            return c
    else:
        return c


@hipy.compiled_function
def fn_fillintimes():
    print(fillintimes(not_constant(10.0), not_constant(""), not_constant("20")))
    print(fillintimes(not_constant(10.0), not_constant("5"), not_constant("20")))
    print(fillintimes(not_constant(10.0), not_constant("0"), not_constant("20")))
    print(fillintimes(not_constant(10.0), not_constant("-3"), not_constant("20")))


def test_fillintimes():
    check_prints(fn_fillintimes, """
10.0
20.0
10.0
10.0
""")
