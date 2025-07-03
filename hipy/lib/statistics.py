__HIPY_MODULE__ = "statistics"

import hipy
import sys
import hipy.intrinsics as intrinsics

hipy.register(sys.modules[__name__])

@hipy.compiled_function
def stdev(l):
    n = len(l)
    if n <= 1:
        return 0.0
    else:
        mean_ = sum(l) / n
        res= sum([(x - mean_) ** 2 for x in l]) / (n - 1)
        return res
@hipy.compiled_function
def mean(l):
    #todo: implement proper
    return sum(l) / len(l)