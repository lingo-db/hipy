import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.lib.builtins import _concrete_dict
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_concrete_dict():
    d = {"a": 1, "b": 2}

    print(d["a"])
    print(d["b"])
    d['c'] = 3
    d["a"] = "hello"
    print(d['c'])
    print(d.setdefault('c',4))
    print(d.setdefault('d',4))
    print(d)


def test_concrete_dict():
    check_prints(fn_concrete_dict, """1
2
3
3
4
{'a': 'hello', 'b': 2, 'c': 3, 'd': 4}
""")


@hipy.compiled_function
def fn_dict():
    d = not_constant({"a": 1, "b": 2})

    d['c'] = 3
    print(d["a"])
    print(d["b"])
    print(d['c'])
    print(d.setdefault('c',4))
    print(d.setdefault('d',4))
    print(d["c"], d["d"])


def test_dict():
    check_prints(fn_dict, """1
2
3
3
4
3 4
""")


@hipy.compiled_function
def fn_dict_in_loop():
    d = {}
    for i in range(11):
        l =d.setdefault(i, [])
        l.append(i*i)
    print(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10])

def test_dict_in_loop():
    check_prints(fn_dict_in_loop, """[0] [1] [4] [9] [16] [25] [36] [49] [64] [81] [100]""")


@hipy.compiled_function
def fn_dict_concrete_non_const():
    d = {}
    d[not_constant("a")] = 1
    print(intrinsics.isa(d,_concrete_dict))
    print(d["a"])
    print(intrinsics.isa(d,_concrete_dict))


def test_dict_concrete_non_const():
    check_prints(fn_dict_concrete_non_const, """
True
1
False""")