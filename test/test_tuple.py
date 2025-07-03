import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_tuple_(t):
    print(t[0])
    print(t[1])
    print(t[2])
    print(t)


@hipy.compiled_function
def fn_tuple():
    fn_tuple_((1, 2, 3))
    fn_tuple_(not_constant((1, 2, 3)))


def test_tuple():
    check_prints(fn_tuple, """1
2
3
(1, 2, 3)
1
2
3
(1, 2, 3)
""")


@hipy.compiled_function
def fn_tuple_constructor():
    print((42, 43))
    print((True, False))
    print((False, True))
    print((0.1, 0.2))


def test_tuple_constructor():
    check_prints(fn_tuple_constructor, """
(42, 43)
(True, False)
(False, True)
(0.1, 0.2)
""")


@hipy.compiled_function
def fn_to_list():
    print(list((1, 2, 3, 4)))


def test_to_list():
    check_prints(fn_to_list, """
[1, 2, 3, 4]
""")

@hipy.compiled_function
def fn_to_python():
    t = (1, 2, 3)
    p = intrinsics.to_python(t)
    print(p)
    if intrinsics.isa(t, tuple):
        print("t is a tuple")
    print(t)

def test_to_python():
    check_prints(fn_to_python, """
(1, 2, 3)
t is a tuple
(1, 2, 3)
""")



@hipy.compiled_function
def fn_unpack():
    t = (1, 2, 3)
    x,y,z = t
    print(x)
    print(y)
    print(z)

def test_unpack():
    check_prints(fn_unpack, """
1
2
3
""")

@hipy.compiled_function
def fn_len():
    print(len((1, 2, 3)))

def test_len():
    check_prints(fn_len, """
3
""")