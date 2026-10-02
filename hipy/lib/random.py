__HIPY_MODULE__ = "random"

import hipy
import sys
import hipy.intrinsics as intrinsics

hipy.register(sys.modules[__name__])


# uniformly distributed in [0, 1); nondeterministic (per-thread generator in
# the runtime, no seeding)
@hipy.compiled_function
def random():
    return intrinsics.call_builtin("random.random", float, [])
