import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.lib.builtins import _const_bool, _const_int
from hipy.test_utils import not_constant


global_str_const=hipy.global_const("hello")

@hipy.compiled_function
def fn_print():
    print(global_str_const)

def test_print():
    check_prints(fn_print, """
hello
""")