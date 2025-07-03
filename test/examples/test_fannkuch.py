import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
from math import factorial as fact
import hipy.lib.math
@hipy.compiled_function
def perm(n, i):
    p = [0] * n

    for k in range(n):
        f = fact(n - 1 - k)
        p[k] = i //f
        i = i % f


    for k in range(n - 1, -1, -1):
        for j in range(k - 1, -1, -1):
            if p[j] <= p[k]:
                p[k] += 1

    return p

@hipy.compiled_function
def fannkuch(n):
    max_flips = 0

    for idx in range(fact(n)):
        p = perm(n, idx)
        flips = 0
        k = p[0]

        while k:
            i = 0
            j = k
            while i < j:
                p[i], p[j] = p[j], p[i]

                i += 1
                j -= 1
            k = p[0]
            flips += 1

        max_flips = max(flips, max_flips)
    return max_flips


@hipy.compiled_function
def fn_fannkuch():
    r=fannkuch(not_constant(4))
    print(r)

def test_fannkuch():
    check_prints(fn_fannkuch, """4""")
