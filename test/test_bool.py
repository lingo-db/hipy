import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant

EXPECTED = """
True
False
False
False
True
True
True
False
False
True"""


@hipy.compiled_function
def fn_bool_ops(t, f):
    print(t and t)
    print(t and f)
    print(f and t)
    print(f and f)
    print(t or t)
    print(t or f)
    print(f or t)
    print(f or f)
    print(not t)
    print(not f)


@hipy.compiled_function
def fn_bool():
    t = not_constant(True)
    f = not_constant(False)
    fn_bool_ops(t, f)


def test_bool():
    check_prints(fn_bool, EXPECTED)


@hipy.compiled_function
def fn_bool_const():
    fn_bool_ops(True, False)


def test_bool_const():
    check_prints(fn_bool_const, EXPECTED)


@hipy.compiled_function
def fn_bool_py():
    fn_bool_ops(intrinsics.to_python(True), intrinsics.to_python(False))


def test_bool_py():
    check_prints(fn_bool_py, EXPECTED)


@hipy.compiled_function
def fn_bool_conversions():
    print(bool(not_constant([])))
    print(bool(not_constant([1])))

def test_bool_conversions():
    check_prints(fn_bool_conversions, """
False
True
""")
