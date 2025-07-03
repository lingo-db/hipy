from hipy.opt.pattern_rewriter import PatternRewriter,RewritePattern
import hipy.ir as ir


class RecordGetPattern(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.RecordGet(member=member, record=record):
                match record.producer:
                    case ir.MakeRecord(values=values):
                        for m,v in values.items():
                            if m== member:
                                rewriter.replace_with_value(op, v)
                                return True
        return False



def canonicalize(module: ir.Module):
    rewriter = PatternRewriter([RecordGetPattern()], module)
    rewriter.rewrite()
    return module