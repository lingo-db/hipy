
import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np

@hipy.compiled_function
def centered_difference_range2d(u, D, dx=1.):
    m, n = u.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            D[i, j] = (u[i + 1, j] + u[i, j + 1] + u[i - 1, j] + u[i, j - 1] - 4.0 * u[i, j]) / dx ** 2
    return D

@hipy.compiled_function
def fn():
    N = 10
    u1 = np.random.rand(N * N)
    dx = 1.5

    u2c = u1.reshape((N, N))
    D2c = np.zeros_like(u2c)
    res=centered_difference_range2d(u2c, D2c, dx)
    print(len(str(res).split(".")))

def test_centered_difference_range2d():
    check_prints(fn, """101""")
