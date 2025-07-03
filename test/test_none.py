import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_none():
    print(None)
    print(not_constant(None))


def test_none():
    check_prints(fn_none, """
None
None
""")


@hipy.compiled_function
def fn_none_is():
    print(None is None)
    print(None is not None)
    print(1 is None)
    print(1 is not None)
    print(None is 1)
    print(None is not 1)


def test_none_is():
    check_prints(fn_none_is, """
True
False
False
True
False
True
""")


@hipy.compiled_function
def to_python():
    print(intrinsics.to_python(None))

def test_to_python():
    check_prints(to_python, """
None
""")

