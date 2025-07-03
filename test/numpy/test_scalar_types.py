import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np

@hipy.compiled_function
def fn_construct():
    print(np.int64(42))
    print(np.int64(42.0))
    print(np.float64(42.0))
    print(np.float64(42))

def test_construct():
    check_prints(fn_construct, """
42
42
42.0
42.0
""")

@hipy.compiled_function
def fn_numeric_ops():
    print(np.int64(1)+np.int64(1))

def test_numeric_ops():
    check_prints(fn_numeric_ops, """
2
""")

@hipy.compiled_function
def fn_cast_to_python():
    print(int(np.int64(42)))
    print(float(np.float64(42.0)))
    print(int(np.float64(42.0)))
    print(float(np.int64(42)))

def test_cast_to_python():
    check_prints(fn_cast_to_python, """
42
42.0
42
42.0
""")