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