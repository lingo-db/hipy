import collections
import hipy.lib.collections
from hipy import intrinsics
import hipy.interpreter

@hipy.compiled_function
def fn_named_tuple():
    P = collections.namedtuple("P",["x","y"])
    p = P(1,2)


    print("p[0]",p[0])
    print("p[1]",p[1])
    print("p.x",p.x)
    print("p.y",p.y)
    print("str",p)
    print("py",intrinsics.to_python(p))

def test_named_tuple():
    hipy.interpreter.check_prints(fn_named_tuple, """
p[0] 1
p[1] 2
p.x 1
p.y 2
str P(x=1, y=2)
py P(x=1, y=2)
""")


from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_namedtuple_kwargs():
    # namedtuple constructor accepts kwargs (the __create__ merges args+kwargs by field).
    P = collections.namedtuple("P", ["a", "b", "c"])
    p = P(a=1, b=2, c=3)
    print(p.a)
    print(p.b)
    print(p.c)
    # Mixed positional + kwargs should also work.
    q = P(10, c=30, b=20)
    print(q.a, q.b, q.c)


def test_namedtuple_kwargs():
    hipy.interpreter.check_prints(fn_namedtuple_kwargs, """
1
2
3
10 20 30
""")


@hipy.compiled_function
def fn_namedtuple_heterogeneous():
    # Heterogeneous field types propagate into NamedTupleType element_types.
    Row = collections.namedtuple("Row", ["name", "age", "score"])
    r = Row(not_constant("alice"), not_constant(30), not_constant(95.5))
    print(r.name)
    print(r.age)
    print(r.score)
    print(r)


def test_namedtuple_heterogeneous():
    hipy.interpreter.check_prints(fn_namedtuple_heterogeneous, """
alice
30
95.5
Row(name='alice', age=30, score=95.5)
""")


@hipy.compiled_function
def fn_namedtuple_in_loop():
    # Using namedtuple inside a loop — each instance shares the cls.
    Pt = collections.namedtuple("Pt", ["x", "y"])
    total_x = 0
    for i in range(3):
        p = Pt(i, i * 2)
        total_x += p.x
    print(total_x)


def test_namedtuple_in_loop():
    hipy.interpreter.check_prints(fn_namedtuple_in_loop, """
3
""")