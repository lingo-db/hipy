from hipy.opt.pattern_rewriter import PatternRewriter,RewritePattern
import hipy.ir as ir


class InlineFunc(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.Call(name=fn_name, args=args):
                fn = rewriter.module.func(fn_name)
                mapping = {p: a for p, a in zip(fn.args, args)}
                for fn_op in fn.body.ops:
                    match fn_op:
                        case ir.Return(values=[v]):
                            rewriter.replace_with_value(op, mapping[v])
                        case _:
                            fn_op.clone(rewriter.before_current(), mapping)
                return True


def run(module: ir.Module):
    rewriter = PatternRewriter([InlineFunc()], module)
    rewriter.rewrite()
    return module
