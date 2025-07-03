import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.lib.builtins import _const_bool, _const_int
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn_print():
    print(b'hello\x00\x01')
    print(not_constant(b'hello\x00\x01'))


def test_print():
    check_prints(fn_print, """
b'hello\\x00\\x01'
b'hello\\x00\\x01'
""")