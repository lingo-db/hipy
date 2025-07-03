from hipy.interpreter import check_prints
import hipy
@hipy.compiled_function
def fn_print():
    print(42)
    print(True)
    print(False)
    print(0.1)

def test_print():
    check_prints(fn_print,"""
42
True
False
0.1
""")

@hipy.compiled_function
def fn_print_multiple():
    print(42, True, False, 0.1)

def test_print_multiple():
    check_prints(fn_print_multiple,"""
42 True False 0.1
""")