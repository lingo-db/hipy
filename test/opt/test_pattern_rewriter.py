import hipy
from hipy.opt.pattern_rewriter import RewritePattern, PatternRewriter
import hipy.ir as ir
import hipy.compiler
from hipy.test_utils import not_constant


class FoldConstAdd(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="scalar.int.add",
                                args=[ir.SSAValue(producer=ir.Constant(v=a)), ir.SSAValue(producer=ir.Constant(v=b))]):
                rewriter.replace_with(op, ir.Constant, v=a + b, t=op.result.type)
                return True
        return False

class FoldConstSub(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="scalar.int.sub",
                                args=[ir.SSAValue(producer=ir.Constant(v=a)), ir.SSAValue(producer=ir.Constant(v=b))]):
                val=rewriter.create(ir.Constant, v=a - b, t=op.result.type).result
                rewriter.replace_with_value(op,val)
                return True
        return False


def fold_const(module: ir.Module):
    rewriter = PatternRewriter([FoldConstAdd(),FoldConstSub()], module)
    rewriter.rewrite()
    return module


@hipy.compiled_function
def fn_add():
    print(not_constant(1) + not_constant(2) + not_constant(3))


def test_fold_const_add():
    module = hipy.compiler.compile(fn_add, fallback=True)
    module = fold_const(module)
    assert str(module).find("const 6") != -1

@hipy.compiled_function
def fn_sub():
    print(not_constant(6) - not_constant(2) - not_constant(3))


def test_fold_const_add():
    module = hipy.compiler.compile(fn_sub, fallback=True)
    module = fold_const(module)
    assert str(module).find("const 3") != -1
