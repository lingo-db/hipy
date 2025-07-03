
import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.statistics
from statistics import mean,stdev
@hipy.compiled_function
def fn_mean():
    print(mean([1, 2, 3]))
    print(mean([1.0, 2.0, 3.0]))
    print(not_constant(mean([1, 2, 3])))
    print(not_constant(mean([1.0, 2.0, 3.0])))

@hipy.compiled_function
def fn_stdev():
    print(stdev([1, 2, 3]))
    print(stdev([1.0, 2.0, 3.0]))
    print(not_constant(stdev([1, 2, 3])))
    print(not_constant(stdev([1.0, 2.0, 3.0])))

def test_mean():
    check_prints(fn_mean, """
2.0
2.0
2.0
2.0""")

def test_stdev():
    check_prints(fn_stdev, """
1.0
1.0
1.0
1.0""")

