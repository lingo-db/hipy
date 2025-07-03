import hipy
import hipy.opt.eager_free as eager_free
import hipy.ir as ir
import hipy.compiler
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import re


@hipy.compiled_function
def fn():
    a = not_constant(1)
    b = not_constant(2)
    print(a + b)

    print(a + 3)
    print("foo")


def test_run():
    module = hipy.compiler.compile(fn, fallback=True)
    print(module)
    eager_free.run(module)
    print(module)
    c1_id = None
    c2_id = None
    c3_id = None
    stage = 0
    for op in module.func("fn").body.ops:
        match op:
            case ir.Constant(v=1, result=ir.SSAValue(id=id)):
                c1_id = id
                assert stage == 0
                stage = 1
            case ir.Constant(v=2, result=ir.SSAValue(id=id)):
                assert stage == 1
                c2_id = id
                stage = 2
            case ir.CallBuiltin(name="scalar.int.add", args=[ir.SSAValue(id=left_id), ir.SSAValue(id=right_id)]):
                if stage == 2:
                    assert left_id == c1_id and right_id == c2_id
                    stage = 3
                elif stage == 5:
                    assert left_id == c1_id and right_id == c3_id
                    stage = 6
            case ir.Free(v=ir.SSAValue(id=id)):
                if stage == 3:
                    assert id == c2_id
                    stage = 4

            case ir.Constant(v=3, result=ir.SSAValue(id=id)):
                c3_id = id
                assert stage == 4
                stage = 5
