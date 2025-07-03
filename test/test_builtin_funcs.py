import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant

@hipy.compiled_function
def fn_min():
    print(min([1,2,3]))
    print(min(1,2,3))
    print(min(not_constant([1,2,3])))
    print(min(not_constant(1),not_constant(2),not_constant(3)))


def test_min():
    check_prints(fn_min, """
1
1
1
1""")