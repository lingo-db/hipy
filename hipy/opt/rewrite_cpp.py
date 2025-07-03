import abc
import textwrap
from typing import Dict, List, Any, Tuple

from hipy.cppbackend import cppir
from hipy.opt.pattern_rewriter import PatternRewriter, RewritePattern
import hipy.ir as ir



class RewriteIterRange(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="range.iter", args=[func, read_only, iter_vals,start, end,step]):
                match func.producer:
                    case ir.FunctionRef(name=funcname, closure=closure):
                        pass
                    case _:
                        return False
                unpacked = {k: rewriter.create(ir.RecordGet,iter_vals.type.member_type(k), iter_vals, k).result for k,v in iter_vals.type.members}
                iter_range=rewriter.create( cppir.IterRange, unpacked,iter_vals.type, start, end, step)
                packed = ir.MakeRecord(iter_range.iter_block, iter_vals.type, iter_range.iter_val_vars).result
                r=ir.Call(iter_range.iter_block, funcname,[read_only,packed,iter_range.iter_var]+([closure] if closure is not None else []),func.type.res_type).result
                r_unpacked=[ir.RecordGet(iter_range.iter_block,func.type.res_type.member_type(k), r, k).result for k,v in func.type.res_type.members]
                ir.Yield(iter_range.iter_block, r_unpacked)
                rewriter.update_uses(iter_range)
                rewriter.replace_with_value(op, iter_range.result)
                return True


class RewriteIterWhile(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="while.iter", args=[cond_func,func, read_only, iter_vals]):
                match func.producer:
                    case ir.FunctionRef(name=funcname, closure=closure):
                        pass
                    case _:
                        return False
                match cond_func.producer:
                    case ir.FunctionRef(name=cond_func_name, closure=cond_closure):
                        pass
                    case _:
                        return False
                iter_while=rewriter.replace_with(op, cppir.WhileIter,read_only, iter_vals)
                cond = ir.Call(iter_while.cond_block, cond_func_name,[read_only,iter_while.iter_val_var]+([cond_closure] if cond_closure is not None else []),ir.bool).result
                ir.Yield(iter_while.cond_block, [cond])
                r=ir.Call(iter_while.iter_block, funcname,[read_only,iter_while.iter_val_var]+([closure] if closure is not None else []),func.type.res_type).result
                ir.Yield(iter_while.iter_block, [r])
                return True

class FusePyMethodCall(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.PythonCall(callable=callable, args=args, kw_args=[]):
                match callable.producer:
                    case ir.PyGetAttr(name=method, on=on):
                        pass
                    case _:
                        return False
                if len(rewriter.uses[callable])>1:
                    return False
                #print("fusing!")

                str_const=rewriter.create(ir.Constant, method, ir.string).result
                py_str_const=rewriter.create(ir.CallBuiltin,"scalar.string.to_python", [str_const],ir.string).result

                rewriter.replace_with(op, cppir.PyMethodCall, on,py_str_const , args)
                rewriter.remove(callable.producer)
                return True


def rewrite(module: ir.Module):
    rewriter = PatternRewriter(
        [RewriteIterRange(),RewriteIterWhile()], module)
    rewriter.rewrite()
    return module
