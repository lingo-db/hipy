import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant

def foo(l):
    l[0][0][0] = 42

@hipy.compiled_function
def fn_list_alias():
    l1 = not_constant([1])
    l2 = not_constant([l1])
    l3 = not_constant([l2])
    foo(l3)
    print(l3[0][0][0])
    print(l1[0])

def test_list_alias():
    check_prints(fn_list_alias, """42
42""")