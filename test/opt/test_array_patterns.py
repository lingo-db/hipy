import hipy
import hipy.opt.inline as inline
import hipy.opt.array_patterns as array_patterns
import hipy.ir as ir
import hipy.compiler
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np
import hipy.interpreter


@hipy.compiled_function
def fn():
    arr= np.ones(10, dtype=np.int64)
    print(arr*2*not_constant(4)+arr*not_constant(3))

def test_run():
    module = hipy.compiler.compile(fn, fallback=False)
    assert str(module).count("array.apply_scalar") == 3 and str(module).count("array.binary_op") == 1
    array_patterns.canonicalize_array_ops(module)
    assert str(module).count("array.apply_scalar") == 0 and str(module).count("array.binary_op") == 0 and str(module).count("array.compute") == 4
    array_patterns.fuse_array_ops(module)
    assert str(module).count("array.compute") == 1

    interpreter_result = hipy.cppbackend.run("fn", module)
    assert interpreter_result==('[11 11 11 11 11 11 11 11 11 11]\n', '', 0)




