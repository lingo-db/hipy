import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np

@hipy.compiled_function
def fn_sin():
    print(np.sin(0.0))
    print(np.sin(1.0))
    print(np.sin(2))
    print(np.sin(np.ones(3)))

def test_sin():
    check_prints(fn_sin, """
0.0
0.8414709848078965
0.9092974268256817
[0.84147098 0.84147098 0.84147098]
""")


@hipy.compiled_function
def fn_cos_sqrt_exp_log():
    # Cover the cos/sqrt/exp/log scalar-&-array overloads (all via _float_function).
    print(np.cos(0.0))
    print(np.sqrt(4.0))
    print(np.log(1.0))
    print(np.exp(0.0))
    print(np.sqrt(np.ones(3) * 9.0))


def test_cos_sqrt_exp_log():
    check_prints(fn_cos_sqrt_exp_log, """
1.0
2.0
0.0
1.0
[3. 3. 3.]
""")


@hipy.compiled_function
def fn_arcsin():
    # np.arcsin — the arcsin branch of _float_function.
    print(np.arcsin(0.0))
    print(np.arcsin(1.0))


def test_arcsin():
    check_prints(fn_arcsin, """
0.0
1.5707963267948966
""")