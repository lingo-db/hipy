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


@hipy.compiled_function
def fn_bool_int_cast():
    # bool.__int__ is defined but not previously exercised: verify 1/0 come out.
    print(int(not_constant(True)))
    print(int(not_constant(False)))
    # Arithmetic with an int should also succeed via the bool -> int coercion.
    print(not_constant(True) + 1)
    print(not_constant(False) + 1)


def test_bool_int_cast():
    check_prints(fn_bool_int_cast, """
1
0
2
1
""")


@hipy.compiled_function
def fn_bool_truthiness_values():
    # bool(...) over various runtime values exercises __bool__ on each type.
    print(bool(not_constant(0)))
    print(bool(not_constant(5)))
    print(bool(not_constant(0.0)))
    print(bool(not_constant(1.5)))
    print(bool(not_constant("")))
    print(bool(not_constant("a")))


def test_bool_truthiness_values():
    check_prints(fn_bool_truthiness_values, """
False
True
False
True
False
True
""", fallback=True)


@hipy.compiled_function
def fn_bool_topython_roundtrip():
    # Round-trip a runtime bool through pyobj and back via the fallback path.
    t = not_constant(True)
    p = intrinsics.to_python(t)
    print(p)
    if intrinsics.isa(p, object):
        print("is pyobj")


def test_bool_topython_roundtrip():
    check_prints(fn_bool_topython_roundtrip, """
True
is pyobj
""")
