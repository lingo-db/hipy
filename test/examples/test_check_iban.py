import hipy
import hipy.compiler
import hipy.interpreter
from hipy.test_utils import not_constant
import hipy.intrinsics as intrinsics
import hipy.lib.numpy
import numpy as np

@hipy.compiled_function
def check_iban(iban):
    iban = iban.replace(" ", "").replace("-", "").replace(".", "")
    iban = iban.upper()
    iban = iban[4:] + iban[:4]
    r = 0
    for c in iban:
        if c.isdigit():
            c = int(c)
            r *= 10
        else:
            c = ord(c) - ord("A") + 10
            r *= 100
        r += c
        r %= 97
    return r % 97 == 1


@hipy.compiled_function
def fn():
    print(check_iban("DE91100000000123456789"))
    print(check_iban("DE91100000000123456788"))
    print(check_iban(not_constant("DE91100000000123456789")))
    print(check_iban(not_constant("DE91100000000123456788")))

def test_iban():
    hipy.interpreter.check_prints(fn, """
True
False
True
False
""", debug=False)