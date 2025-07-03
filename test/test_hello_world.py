import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant

@hipy.compiled_function
def fn_hello_world():
    print("hello"+" "+"world")
    name = not_constant("world")
    print("hello "+name)


def test_hello_world():
    check_prints(fn_hello_world, """
hello world
hello world""")
