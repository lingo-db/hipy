import hipy
import hipy.opt.eliminate_dead_code as eliminate_dead_code
import hipy.ir as ir
import hipy.compiler
from hipy.test_utils import not_constant


@hipy.compiled_function
def fn():
    not_constant(1)
    not_constant((1,2,3,4))
    print("foo")

def test_run():
    module = hipy.compiler.compile(fn, fallback=True)
    print(module)
    eliminate_dead_code.run(module)
    print(module)
    assert str(module).find("const 1") == -1 and str(module).find("record") == -1