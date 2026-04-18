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


@hipy.compiled_function
def fn_dict_contains():
    # dict.__contains__ → dict.contains builtin.
    d = not_constant({"a": 1, "b": 2})
    print("a" in d)
    print("c" in d)
    print("b" not in d)


def test_dict_contains():
    check_prints(fn_dict_contains, """
True
False
False
""")


@hipy.compiled_function
def fn_dict_len():
    # dict.__len__ → dict.length builtin.
    d = not_constant({"a": 1, "b": 2, "c": 3})
    print(len(d))


def test_dict_len():
    check_prints(fn_dict_len, """
3
""")


@hipy.compiled_function
def fn_dict_get():
    # dict.get exercises `key in self` + default-fallback path distinct from setdefault.
    d = not_constant({"a": 1, "b": 2})
    print(d.get("a", 99))
    print(d.get("x", 99))
    # After get with missing key, the dict should not be mutated.
    print("x" in d)


def test_dict_get():
    check_prints(fn_dict_get, """
1
99
False
""")


@hipy.compiled_function
def fn_dict_iter():
    # Iterating a dict yields keys (via dict.iter_keys builtin).
    d = not_constant({"a": 1, "b": 2, "c": 3})
    total = 0
    for k in d:
        total += d[k]
    print(total)


def test_dict_iter():
    check_prints(fn_dict_iter, """
6
""")


@hipy.compiled_function
def fn_dict_items():
    # dict.items() returns a _items view iterable yielding tuples.
    d = not_constant({"a": 1, "b": 2})
    total = 0
    for k, v in d.items():
        total += v
    print(total)


def test_dict_items():
    check_prints(fn_dict_items, """
3
""")


@hipy.compiled_function
def fn_dict_topython_roundtrip():
    # dict.__topython__ iterates + builds python dict via python.create_dict.
    # Iteration order of HiPy dict is unspecified (does not preserve insertion
    # order), so probe by key membership rather than printing the whole dict.
    d = not_constant({"a": 1, "b": 2})
    p = intrinsics.to_python(d)
    if intrinsics.isa(p, object):
        print("is pyobj")
    print(p["a"], p["b"])


def test_dict_topython_roundtrip():
    check_prints(fn_dict_topython_roundtrip, """
is pyobj
1 2
""", fallback=True)