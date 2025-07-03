import hipy
import hipy.opt.inline as inline
import hipy.opt.eliminate_dead_symbols as eliminate_dead_symbols
import hipy.ir as ir
import hipy.compiler
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant


def test_inline():
    module = ir.Module()
    fn = ir.Function(module, "fn", [], ir.void)
    ir.Return(fn.body, [
        ir.Call(fn.body, "add", [ir.Constant(fn.body, 1,ir.int).result, ir.Constant(fn.body, 2, ir.int).result], ir.int).result])
    add = ir.Function(module, "add", [ir.int,ir.int], ir.int)
    ir.Return(add.body, [ir.CallBuiltin(add.body, "scalar.int.add", [add.args[0], add.args[1]], ir.int).result])
    print(module)
    eliminate_dead_symbols.run(module, "fn")
    assert "func add" in str(module) and "func fn" in str(module)
    print(module)
    inline.run(module)
    print(module)
    eliminate_dead_symbols.run(module, "fn")
    print(module)
    assert "func add" not in str(module)  and "func fn" in str(module)