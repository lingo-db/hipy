import pytest

import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.lib.builtins import _const_bool, _const_int
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_comparisons():
    const_eq_t="a" == "a"
    const_eq_f="a" == "b"
    print("== :",const_eq_t,intrinsics.isa(const_eq_t,_const_bool))
    print("== :",const_eq_f,intrinsics.isa(const_eq_f,_const_bool))
    print("==:","a" == not_constant("a"))
    print("==:","a" == not_constant("b"))

    const_ne_t = "a" != "a"
    const_ne_f = "a" != "b"
    print("!= :", const_ne_t, intrinsics.isa(const_ne_t, _const_bool))
    print("!= :", const_ne_f, intrinsics.isa(const_ne_f, _const_bool))
    print("!=:", "a" != not_constant("a"))
    print("!=:", "a" != not_constant("b"))

    const_lt_t = "a" < "a"
    const_lt_f = "a" < "b"
    print("< :", const_lt_t, intrinsics.isa(const_lt_t, _const_bool))
    print("< :", const_lt_f, intrinsics.isa(const_lt_f, _const_bool))
    print("<:", "a" < not_constant("a"))
    print("<:", "a" < not_constant("b"))

    const_le_f = "b" <= "a"
    const_le_t = "b" <= "b"
    const_le_t2 = "b" <= "c"

    print("<= :", const_le_f, intrinsics.isa(const_le_f, _const_bool))
    print("<= :", const_le_t, intrinsics.isa(const_le_t, _const_bool))
    print("<= :", const_le_t2, intrinsics.isa(const_le_t2, _const_bool))
    print("<=:", "b" <= not_constant("a"))
    print("<=:", "b" <= not_constant("b"))
    print("<=:", "b" <= not_constant("c"))

    const_gt_t = "a" > "a"
    const_gt_f = "b" > "a"
    print("> :", const_gt_t, intrinsics.isa(const_gt_t, _const_bool))
    print("> :", const_gt_f, intrinsics.isa(const_gt_f, _const_bool))
    print(">:", "a" > not_constant("a"))
    print(">:", "b" > not_constant("a"))

    const_ge_f = "a" >= "b"
    const_ge_t = "a" >= "a"
    const_ge_t2 = "b" >= "a"
    print(">= :", const_ge_f, intrinsics.isa(const_ge_f, _const_bool))
    print(">= :", const_ge_t, intrinsics.isa(const_ge_t, _const_bool))
    print(">= :", const_ge_t2, intrinsics.isa(const_ge_t2, _const_bool))
    print(">=:", "a" >= not_constant("a"))
    print(">=:", "a" >= not_constant("b"))
    print(">=:", "b" >= not_constant("a"))

    const_in_t = "a" in "abc"
    const_in_f = "d" in "abc"
    print("in :", const_in_t, intrinsics.isa(const_in_t, _const_bool))
    print("in :", const_in_f, intrinsics.isa(const_in_f, _const_bool))
    print("in:", "a" in not_constant("abc"))
    print("in:", "d" in not_constant("abc"))



def test_comparisons():
    check_prints(fn_comparisons, """
== : True True
== : False True
==: True
==: False
!= : False True
!= : True True
!=: False
!=: True
< : False True
< : True True
<: False
<: True
<= : False True
<= : True True
<= : True True
<=: False
<=: True
<=: True
> : False True
> : True True
>: False
>: True
>= : False True
>= : True True
>= : True True
>=: True
>=: False
>=: True
in : True True
in : False True
in: True
in: False
""")

@hipy.compiled_function
def fn_ops():
    print("+:", "a" + "b")
    print("+:", "a" + not_constant("b"))
    const_len = len("abc")
    print("len:", const_len, intrinsics.isa(const_len, _const_int))
    print("len:",len("abc"))
    print("lower:","aBc".lower())
    print("upper:","aBc".upper())
    print(",".join(["a", "b", "c"]))

def test_ops():
    check_prints(fn_ops, """
+: ab
+: ab
len: 3 True
len: 3
lower: abc
upper: ABC
a,b,c
""")



@hipy.compiled_function
def fn_string_find():
    print("abcdef".find("cd"))
    print("abcdef".find("ab",1))
    print("abcdef".find("ab",0,2))
    print("abcdef".rfind("def"))
    print("abcdef".rfind("def",3,6))
    print("abcdef".rfind("def",3,4))
    print("2 bds , 1".rfind(","))

def test_string_find():
    check_prints(fn_string_find, """2
-1
0
3
3
-1
6
""")

@hipy.compiled_function
def fn_string_replace():
    print("abcdef".replace("cd","xy"))
    print("abcdefcd".replace("cd","xy"))

def test_string_replace():
    check_prints(fn_string_replace, """abxyef
abxyefxy""")


@hipy.compiled_function
def fn_string_slice():
    print("abcdef"[1:3])
    print("abcdef"[1:])
    print("abcdef"[:-1])
    print("abcdef"[-2:])
    print("abcdef"[0])

def test_string_slice():
    check_prints(fn_string_slice, """
bc
bcdef
abcde
ef
a
    """)

@hipy.compiled_function
def fn_string_iter():
    s = "abcdef"
    for c in s:
        print(c)
    for c in not_constant(s):
        print(c)

def test_string_iter():
    check_prints(fn_string_iter, """
a
b
c
d
e
f
a
b
c
d
e
f
""")


@hipy.compiled_function
def fn_string_partition():
    print("abcdcdef".partition("cd"))
    print("abcdcdef".partition("xy"))
    print("abcdcdef".rpartition("cd"))
    print("abcdcdef".rpartition("xy"))

def test_string_partition():
    check_prints(fn_string_partition, """
('ab', 'cd', 'cdef')
('abcdcdef', '', '')
('abcd', 'cd', 'ef')
('', '', 'abcdcdef')
""")


@hipy.compiled_function
def fn_string_split():
    print("a,b,c".split(","))
    print("a,b,c".split(",",1))
def test_string_split():
    check_prints(fn_string_split, """
['a', 'b', 'c']
['a', 'b,c']
""")


@hipy.compiled_function
def fn_string_ord():
    print(ord("a"))
    print(ord(not_constant("a")))

def test_string_ord():
    check_prints(fn_string_ord, """
97
97
""")

@hipy.compiled_function
def fn_string_count():
    print("ababc".count("ab"))
    print("ababc".count("ab",1))
    print("ababc".count("ab",0,4))

def test_string_count():
    check_prints(fn_string_count, """2
1
2
""")

@hipy.compiled_function
def fn_string_split_noparams():
    print("a b c".split())
    print(" a  b   c  ".split())
    print("a\nb\nc".split())
    print("a\tb\tc".split())
def test_string_split_noparams():
    check_prints(fn_string_split_noparams, """
['a', 'b', 'c']
['a', 'b', 'c']
['a', 'b', 'c']
['a', 'b', 'c']
""")


@hipy.compiled_function
def fn_string_strip():
    # str.strip / str.rstrip route to scalar.string.strip / rstrip.
    print(not_constant("  hello  ").strip())
    print(not_constant("  hello  ").rstrip())
    print(not_constant("\t\nfoo\r\n").strip())


def test_string_strip():
    check_prints(fn_string_strip, """
hello
  hello
foo
""")


@hipy.compiled_function
def fn_string_iadd():
    # str.__iadd__ falls through to __add__ (scalar.string.concatenate).
    s = not_constant("abc")
    s += "de"
    print(s)
    s += not_constant("fg")
    print(s)


def test_string_iadd():
    check_prints(fn_string_iadd, """
abcde
abcdefg
""")


@hipy.compiled_function
def fn_string_isdigit():
    # isdigit on single char via ord comparisons.
    print("5".isdigit())
    print(not_constant("5").isdigit())
    print(not_constant("a").isdigit())


def test_string_isdigit():
    check_prints(fn_string_isdigit, """
True
True
False
""")


@hipy.compiled_function
def fn_string_isascii():
    # isascii iterates and checks ord(c) > 127.
    print(not_constant("hello").isascii())
    print(not_constant("héllo").isascii())
    print(not_constant("").isascii())


@pytest.mark.xfail(reason="str.isascii returns True for non-ASCII strings — scalar.string.iter or ord() on abstract char doesn't report >127 for multi-byte UTF-8 chars")
def test_string_isascii():
    check_prints(fn_string_isascii, """
True
False
True
""")


@hipy.compiled_function
def fn_string_slice_with_step():
    # str.__getitem__ slice with step goes through range + join.
    s = not_constant("abcdef")
    print(s[::2])
    print(s[1::2])


def test_string_slice_with_step():
    check_prints(fn_string_slice_with_step, """
ace
bdf
""")


@hipy.compiled_function
def fn_string_slice_negative_step():
    # Reversed slice. str.__getitem__ normalizes start/stop but does not handle
    # default values for negative step — known limitation.
    s = not_constant("abcdef")
    print(s[::-1])


@pytest.mark.xfail(reason="str.__getitem__ slice default start/stop (0, len) produces empty range when step is negative — negative-step slicing broken")
def test_string_slice_negative_step():
    check_prints(fn_string_slice_negative_step, """
fedcba
""")


@hipy.compiled_function
def fn_string_negative_index():
    # Runtime-abstract string indexed with negative index currently NOT supported
    # directly (no normalization), but [-2:] slice is. Verify slice-with-negatives on abstract.
    s = not_constant("abcdef")
    print(s[-3:])
    print(s[-5:-2])
    print(s[:-3])


def test_string_negative_index():
    check_prints(fn_string_negative_index, """
def
bcd
abc
""")


@hipy.compiled_function
def fn_string_topython_roundtrip():
    # Round-trip abstract string through pyobj (scalar.string.to_python +
    # from_python) via str.__create__ object path.
    s = not_constant("hello")
    p = intrinsics.to_python(s)
    print(p)
    if intrinsics.isa(p, object):
        print("is pyobj")
    # Cast back to native str: object_to_str path invokes builtins.str()
    # + scalar.string.from_python.
    back = str(p)
    if intrinsics.isa(back, str):
        print("is str")
    print(back)


def test_string_topython_roundtrip():
    check_prints(fn_string_topython_roundtrip, """
hello
is pyobj
is str
hello
""", fallback=True)


@hipy.compiled_function
def fn_string_repr():
    # __hipy__repr__ wraps in single quotes; used when str appears inside a tuple/list print.
    print(repr(not_constant("abc")))
    print((not_constant("a"), not_constant("b")))


def test_string_repr():
    check_prints(fn_string_repr, """
'abc'
('a', 'b')
""")


@hipy.compiled_function
def fn_string_join_abstract_items():
    # str.join iterates and concatenates; exercise with runtime (abstract) strings.
    parts = [not_constant("x"), not_constant("y"), not_constant("z")]
    print(not_constant("-").join(parts))


def test_string_join_abstract_items():
    check_prints(fn_string_join_abstract_items, """
x-y-z
""")


@hipy.compiled_function
def fn_string_split_with_pattern_abstract():
    # scalar.string.split builtin with a runtime separator and maxsplit.
    s = not_constant("a,b,c,d")
    print(s.split(not_constant(",")))
    print(s.split(not_constant(","), 2))


def test_string_split_with_pattern_abstract():
    check_prints(fn_string_split_with_pattern_abstract, """
['a', 'b', 'c', 'd']
['a', 'b', 'c,d']
""")


@hipy.compiled_function
def fn_bytes_len_and_index():
    # bytes.__len__ + bytes.__getitem__(int) via scalar.string.length / at.
    # Note: in HiPy bytes[i] returns a bytes of length 1 (via scalar.string.at),
    # unlike CPython which returns an int.
    b = not_constant(b'hello')
    print(len(b))
    print(b[0])
    print(b[4])


def test_bytes_len_and_index():
    check_prints(fn_bytes_len_and_index, """
5
b'h'
b'o'
""")


@hipy.compiled_function
def fn_bytes_slice():
    # bytes.__getitem__(slice) → scalar.string.substr.
    b = not_constant(b'abcdef')
    print(b[1:4])
    print(b[:3])
    print(b[3:])


def test_bytes_slice():
    check_prints(fn_bytes_slice, """
b'bcd'
b'abc'
b'def'
""")