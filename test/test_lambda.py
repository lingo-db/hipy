import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def apply(x, fn):
    return fn(x)


def apply_python(x, fn):
    return fn(x)


@hipy.compiled_function
def fn_lambda():
    y = 1
    l = lambda x: x + y
    print(apply(2, l))
    y = 2
    print(apply(2, l))


def test_lambda():
    check_prints(fn_lambda, """3
4""")


@hipy.compiled_function
def fn_lambda_python():
    y = 1
    l = lambda x: x + y
    print(apply_python(2, l))
    y = 2
    print(apply_python(2, l))


def test_lambda_python():
    check_prints(fn_lambda_python, """3
4""")


@hipy.compiled_function
def create_add_fn(x):
    return lambda y: x + y


@hipy.compiled_function
def fn_return_lambda():
    l = create_add_fn(42)
    print(apply(2, l))


def test_return_lambda():
    check_prints(fn_return_lambda, """44""")


@hipy.compiled_function
def fn_return_lambda_python():
    l = create_add_fn(42)
    print(apply_python(2, l))


def test_return_lambda_python():
    check_prints(fn_return_lambda_python, """44""")


@hipy.compiled_function
def fn_builtin():
    z= not_constant(3)
    l = lambda x: x+z
    b = intrinsics.bind(l, [int])
    print(intrinsics.call_builtin("test.apply",int,[b,1]))

def test_builtin():
    check_prints(fn_builtin, """4""")

@hipy.compiled_function
def fn_builtin_nested():
    z= not_constant(3)
    l1 = lambda x: x+z
    l2 = lambda x: l1(x)
    l3 = lambda x: l2(x)
    b = intrinsics.bind(l3, [int])
    print(intrinsics.call_builtin("test.apply",int,[b,1]))

def test_builtin_nested():
    check_prints(fn_builtin_nested, """4""")

@hipy.compiled_function
def foo(x):
    return x+3
@hipy.compiled_function
def fn_builtin_fn():
    b = intrinsics.bind(foo, [int])
    print(intrinsics.call_builtin("test.apply",int,[b,1]))

def test_builtin_fn():
    check_prints(fn_builtin_fn, """4""")

def pyfoo(x:int) -> int:
    return x+3

@hipy.compiled_function
def fn_builtin_py():
    b = intrinsics.bind(pyfoo, [int])
    r=intrinsics.call_builtin("test.apply",int,[b,1])
    print(r, intrinsics.isa(r,int))
    b = intrinsics.bind(lambda x:pyfoo(x), [int])
    r=intrinsics.call_builtin("test.apply",int,[b,1])
    print(r, intrinsics.isa(r,int))


def test_builtin_py():
    check_prints(fn_builtin_py, """
4 True
4 True""")