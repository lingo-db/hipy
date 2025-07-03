import hipy
from hipy import intrinsics
from hipy.lib.numpy import ndarray,empty,float64
@hipy.compiled_function
def rand(*args):
    shape=args
    res = empty(shape, float64)
    fill_fn = intrinsics.bind(lambda indices: float64(intrinsics.call_builtin("random.rand", float,[],side_effects=False)), [intrinsics.typeof(shape)])

    intrinsics.call_builtin("array.fill", None, [res, fill_fn])
    return res
