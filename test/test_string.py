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