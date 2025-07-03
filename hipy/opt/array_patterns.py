from hipy.opt.pattern_rewriter import PatternRewriter,RewritePattern
import hipy.ir as ir


class ArrayApplyToArrayCompute(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="array.apply_scalar", args=[array, func]):
                rewriter.replace_with(op, ir.CallBuiltin, name="array.compute", args=[array, func],
                                      ret_type=op.result.type, side_effects=False)
                return True
        return False


class ArrayBinaryOpToArrayCompute(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="array.binary_op", args=[array1, array2, func]):
                rewriter.replace_with(op, ir.CallBuiltin, name="array.compute", args=[array1, array2, func],
                                      ret_type=op.result.type, side_effects=False)
                return True
        return False


def canonicalize_array_ops(module: ir.Module):
    rewriter = PatternRewriter([ArrayApplyToArrayCompute(), ArrayBinaryOpToArrayCompute()], module)
    rewriter.rewrite()
    return module


fused_cntr = 0


class FuseArrayCompute(RewritePattern):

    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="array.compute", args=args1):
                func1 = args1[-1]
                match func1.producer:
                    case ir.FunctionRef(name=func1_name, closure=closure1):
                        pass
                    case _:
                        return False
                arrays = args1[:-1]
                to_fuse = []
                any_useful = False
                for array in arrays:
                    match array.producer:
                        case ir.CallBuiltin(name="array.compute", args=args2):
                            func2 = args2[-1]
                            match func2.producer:
                                case ir.FunctionRef(name=func2_name, closure=closure2):
                                    pass
                                case _:
                                    return False
                            arrays2 = args2[:-1]
                            to_fuse.append((arrays2, (func2_name, closure2)))
                            any_useful = True
                        case _:
                            to_fuse.append(([array], None))
                if any_useful:
                    array_arguments = []
                    func_types = []
                    for arrays, _ in to_fuse:
                        for array in arrays:
                            func_types.append(array.type.element_type)
                            array_arguments.append(array)

                    closure_types = [(f"f{i}", closure.type) for i, (_, (_, closure)) in enumerate(to_fuse) if
                                     closure is not None]
                    closure_vals = {f"f{i}": closure for i, (_, (_, closure)) in enumerate(to_fuse) if
                                    closure is not None}
                    if closure1 is not None:
                        closure_types.append(("f'", closure1.type))
                        closure_vals["f'"] = closure1
                    closure_type = ir.RecordType(closure_types)
                    closure_val = rewriter.create(ir.MakeRecord, closure_type, closure_vals).result
                    global fused_cntr
                    fused_cntr += 1
                    new_func = ir.Function(rewriter.module, f"fused_{fused_cntr}", func_types + [closure_type],
                                           func1.type.res_type)
                    arg_cntr = 0
                    args_for_op = []
                    for i, (arrays, fn) in enumerate(to_fuse):
                        if fn is None:
                            args_for_op.append(new_func.args[arg_cntr])
                            arg_cntr += 1
                        else:
                            fn_name, closure = fn
                            if closure is None:
                                local_closure_val = []
                            else:
                                local_closure_val = [
                                    ir.RecordGet(new_func.body, closure.type, new_func.args[-1], f"f{i}").result]
                            local_args = []
                            for _ in arrays:
                                local_args.append(new_func.args[arg_cntr])
                                arg_cntr += 1
                            args_for_op.append(ir.Call(new_func.body, fn_name, local_args + local_closure_val,
                                                       rewriter.module.func(fn_name).res_type).result)

                    if closure1 is not None:
                        args_for_op.append(ir.RecordGet(new_func.body, closure1.type, new_func.args[-1], f"f'").result)
                    ir.Return(new_func.body,
                              [ir.Call(new_func.body, func1_name, args_for_op, func1.type.res_type).result])
                    func_ref = rewriter.create(ir.FunctionRef, new_func, closure_val).result
                    rewriter.replace_with(op, ir.CallBuiltin, name="array.compute", args=array_arguments + [func_ref],
                                          ret_type=op.result.type, side_effects=False)
                    return True
        return False


def fuse_array_ops(module: ir.Module):
    rewriter = PatternRewriter([FuseArrayCompute()], module)
    rewriter.rewrite()
    return module
