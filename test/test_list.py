import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_print_three(l):
    print(l[0])
    print(l[1])
    print(l[2])

@hipy.compiled_function
def fn_concrete_list():
    l = [1, 2, 3]
    fn_print_three(l)

def test_concrete_list():
    check_prints(fn_concrete_list, """1
2
3""")


@hipy.compiled_function
def fn_list():
    l = not_constant([1, 2, 3])
    fn_print_three(l)

def test_list():
    check_prints(fn_list, """1
2
3""")


@hipy.compiled_function
def fn_list_constructor():
    print([42, 43])
    print([True, False])
    print([False, True])
    print([0.1, 0.2])



def test_list_constructor():
    check_prints(fn_list_constructor, """
[42, 43]
[True, False]
[False, True]
[0.1, 0.2]
""")


@hipy.compiled_function
def fn_list_append():
    l = [42]
    l.append(43)
    l.append(44)
    print(l)
    l = not_constant([42])
    l.append(43)
    l.append(44)
    print(l)


def test_list_append():
    check_prints(fn_list_append, """
[42, 43, 44]
[42, 43, 44]
""")


@hipy.compiled_function
def fn_list_iter_(l):
    x = 1
    for i in l:
        print(i)
        x = x + i
    print(x)

@hipy.compiled_function
def fn_list_iter():
    fn_list_iter_([42, 43, 44])
    fn_list_iter_(not_constant([42, 43, 44]))


def test_list_iter():
    check_prints(fn_list_iter, """
42
43
44
130
42
43
44
130
""")


@hipy.compiled_function
def fn_list_append_different_types():
    l = [42]
    l.append("abc")
    print(l)
    l = not_constant([42])
    l.append("abc")
    print(l)
def test_list_append_different_types():
    check_prints(fn_list_append_different_types, """
[42, 'abc']
[42, 'abc']
""", fallback=True)


@hipy.compiled_function
def fn_empty_list():
    l = []
    for i in not_constant(range(10)):
        l.append(i)
    if intrinsics.isa(l, list):
        print("is a list")
    print(l)


def test_empty_list():
    check_prints(fn_empty_list, """
is a list
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
""")

@hipy.compiled_function
def fn_list_compreshension_(l):
    l1=[i for i in l]
    l2=[i + 1 for i in l]
    l3=[i + 1 for i in l if i % 2 == 1]
    if intrinsics.isa(l1, list):
        print("l1 is a list")
    print(l1)
    if intrinsics.isa(l2, list):
        print("l2 is a list")
    print(l2)
    if intrinsics.isa(l3, list):
        print("l3 is a list")
    print(l3)

@hipy.compiled_function
def fn_list_compreshension():
    lc=[42, 43, 44]
    l=not_constant([42, 43, 44])
    fn_list_compreshension_(lc)
    fn_list_compreshension_(l)

def test_list_compreshension():
    check_prints(fn_list_compreshension, """
l1 is a list
[42, 43, 44]
l2 is a list
[43, 44, 45]
l3 is a list
[44]
l1 is a list
[42, 43, 44]
l2 is a list
[43, 44, 45]
l3 is a list
[44]
""")


@hipy.compiled_function
def fn_for_empty_list():
    l = []
    l2 =[]
    for i in not_constant(range(10)):
        l2.append(l)
    l.append(10)
    print(l2)


def test_for_empty_list():
    check_prints(fn_for_empty_list, """
[[10], [10], [10], [10], [10], [10], [10], [10], [10], [10]]
""",fallback=True)


@hipy.compiled_function
def fn_len():
    print(len([1, 2, 3]))
    print(len(not_constant([1, 2, 3])))

def test_len():
    check_prints(fn_len, """
3
3
""")


@hipy.compiled_function
def fn_narrow():
    l = not_constant([[1]])
    l.append([])
    print(l)
    d = not_constant({"a": [1]})
    d["a"]=[]
    print(d)

def test_narrow():
    check_prints(fn_narrow, """
[[1], []]
{'a': []}
""")


@hipy.compiled_function
def fn_multiply():
    print([1, 2, 3] * 3)
    print(not_constant([1, 2, 3]) * 3)

def test_multiply():
    check_prints(fn_multiply, """
[1, 2, 3, 1, 2, 3, 1, 2, 3]
[1, 2, 3, 1, 2, 3, 1, 2, 3]
""")