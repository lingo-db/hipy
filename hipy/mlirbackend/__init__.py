import builtins
import pathlib
import subprocess
import sys

from lingodbbridge.mlir import ir as mlir
from lingodbbridge.mlir.dialects import func, arith, scf, util, tuples, db, relalg, subop, builtin
import lingodbbridge.mlir._mlir_libs.mlir_init as mlir_init
import lingodbbridge.mlir.extras.types as mlirtypes
import lingodbbridge
import hipy.ir as ir
import lingodb

curr_context = None
curr_module = None


def to_mlir_type(t):
    match t:
        case ir.BoolType():
            return mlir.IntegerType.get_signless(1)

        case ir.IntegerType(width=w):
            return mlir.IntegerType.get_signless(w)
        case ir.FloatType(width=w):
            match w:
                case 32:
                    return mlir.F32Type.get()
                case 64:
                    return mlir.F64Type.get()
                case _:
                    assert False
        case ir.StringType():
            return db.StringType.get(curr_context)
        case ir.IntType():
            return mlirtypes.i64()
        case ir.RecordType(members=members):
            return mlirtypes.TupleType.get_tuple([to_mlir_type(t) for m, t in members], curr_context)
        case ir.FunctionRefType(arg_types=arg_types, res_type=res_type, closure_type=closure_type):
            if closure_type is None:
                return mlirtypes.FunctionType.get([to_mlir_type(t) for t in arg_types], [to_mlir_type(res_type)],
                                                  curr_context)
            else:
                assert False
        case ir.ListType(element_type=elem_type):
            return db.ListType.get(to_mlir_type(elem_type))
        case _:
            assert False
    print(t)
    assert False


def str_attr(s):
    return mlir.StringAttr.get(s)


tmp_col_cntr = 0
tmp_member_cntr = 0


def get_tmp_col_name():
    global tmp_col_cntr
    tmp_col_cntr += 1
    return f"tmp_col_{tmp_col_cntr}"


def get_tmp_member_name():
    global tmp_member_cntr
    tmp_member_cntr += 1
    return f"tmp_member_{tmp_member_cntr}"


helper_fn_cntr = 0


def call(callee, args, mapping):
    match callee.type:
        case ir.FunctionRefType(arg_types=arg_types, res_type=res_type, closure_type=closure_type):
            if closure_type is not None:
                assert False
            else:
                if isinstance(res_type, ir.VoidType):
                    assert False
                else:
                    match callee.producer:
                        case ir.FunctionRef(name=func_name):
                            callOp = func.CallOp([to_mlir_type(res_type)], func_name, args)
                            return callOp.results[0]
                        case _:
                            callOp = func.CallIndirectOp([to_mlir_type(res_type)], mapping[callee], args)
                            return callOp.results[0]
    assert False


def to_mlir_stmt(stmt, mapping):
    global helper_fn_cntr
    match stmt:
        case ir.Return(values=[value]):
            if isinstance(value.type, ir.VoidType):
                func.ReturnOp([])  # returning a void value is equivalent to not returning anything
            else:
                func.ReturnOp([mapping[value]])
        case ir.Constant(result=r, v=v):
            match r.type:
                case ir.BoolType():
                    mapping[r] = arith.ConstantOp(to_mlir_type(r.type), 1 if v else 0).result
                case ir.IntegerType() | ir.FloatType():
                    mapping[r] = arith.ConstantOp(to_mlir_type(r.type), v).result
                case ir.StringType():
                    mapping[r] = db.ConstantOp(to_mlir_type(r.type), str_attr(v)).result
                case ir.ListType(element_type=elem_type):
                    assert False
                    # list = dbpy.CallBuiltin(to_mlir_type(r.type), str_attr("list.create"), []).result
                    # block = ir.Block()
                    # for elem in v:
                    #    constOp = ir.Constant(block, elem, elem_type)
                    #    to_mlir_stmt(constOp, mapping)
                    #    dbpy.CallBuiltin(to_mlir_type(r.type), str_attr("list.append"), [list, mapping[constOp.result]])
                    # mapping[r] = list
                case _:
                    res_type = to_mlir_type(r.type)
                    if (isinstance(res_type, mlir.IntegerType)):
                        mapping[r] = arith.ConstantOp(res_type, v).result
                    else:
                        # mapping[r] = dbpy.ConstantOp(to_mlir_type(r.type), str_attr(str(v))).result
                        assert False
        case ir.CallBuiltin(result=r, name=name, args=args):
            arg_types = [arg.type for arg in args]
            match name, arg_types:
                case "undef", _:
                    mapping[r] = util.UndefOp(to_mlir_type(r.type)).result
                case "scalar.int.add", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.AddIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.add", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.AddIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.sub", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.SubIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.sub", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.SubIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.mul", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.MulIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.mul", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.MulIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.div", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.DivSIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.div", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.DivSIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.mod", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.RemSIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.mod", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.RemSIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.lshift", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.ShLIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.lshift", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.ShLIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.eq", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.eq, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.eq", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.eq, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.neq", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.ne, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.neq", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.ne, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.lt", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.slt, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.lt", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.slt, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.lte", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.sle, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.lte", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.sle, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.gt", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.sgt, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.gt", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.sgt, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.gte", [ir.IntegerType(), ir.IntegerType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.sge, mapping[args[0]], mapping[args[1]]).result
                case "scalar.int.compare.gte", [ir.IntType(), ir.IntType()]:
                    mapping[r] = arith.CmpIOp(arith.CmpIPredicate.sge, mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.add", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.AddFOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.sub", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.SubFOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.mul", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.MulFOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.div", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.DivFOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.compare.eq", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.CmpFOp(arith.CmpFPredicate.OEQ, mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.compare.neq", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.CmpFOp(arith.CmpFPredicate.ONE, mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.compare.lt", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.CmpFOp(arith.CmpFPredicate.OLT, mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.compare.lte", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.CmpFOp(arith.CmpFPredicate.OLE, mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.compare.gt", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.CmpFOp(arith.CmpFPredicate.OGT, mapping[args[0]], mapping[args[1]]).result
                case "scalar.float.compare.gte", [ir.FloatType(), ir.FloatType()]:
                    mapping[r] = arith.CmpFOp(arith.CmpFPredicate.OGE, mapping[args[0]], mapping[args[1]]).result
                case "scalar.bool.and", [ir.BoolType(), ir.BoolType()]:
                    mapping[r] = arith.AndIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.bool.or", [ir.BoolType(), ir.BoolType()]:
                    mapping[r] = arith.OrIOp(mapping[args[0]], mapping[args[1]]).result
                case "scalar.bool.not", [ir.BoolType()]:
                    mapping[r] = arith.XOrIOp(mapping[args[0]],
                                              arith.ConstantOp(to_mlir_type(ir.BoolType()), 1).result).result
                case "scalar.float.from_int", [ir.IntegerType()]:
                    mapping[r] = arith.SIToFPOp(to_mlir_type(r.type), mapping[args[0]]).result
                case "scalar.string.compare.eq", [ir.StringType(), ir.StringType()]:
                    mapping[r] = db.CmpOp(db.DBCmpPredicate.eq, mapping[args[0]], mapping[args[1]]).result
                case "scalar.string.compare.lt", [ir.StringType(), ir.StringType()]:
                    mapping[r] = db.CmpOp(db.DBCmpPredicate.lt, mapping[args[0]], mapping[args[1]]).result
                case "scalar.string.compare.lte", [ir.StringType(), ir.StringType()]:
                    mapping[r] = db.CmpOp(db.DBCmpPredicate.lte, mapping[args[0]], mapping[args[1]]).result
                case "scalar.string.lower", [ir.StringType()]:
                    mapping[r] = db.RuntimeCall(db.StringType.get(curr_context), str_attr("ToLower"),
                                                [mapping[args[0]]]).result
                case "scalar.string.upper", [ir.StringType()]:
                    mapping[r] = db.RuntimeCall(db.StringType.get(curr_context), str_attr("ToUpper"),
                                                [mapping[args[0]]]).result
                case "scalar.string.contains", [ir.StringType(), ir.StringType()]:
                    mapping[r] = db.RuntimeCall(mlirtypes.bool(), str_attr("Contains"),
                                                [mapping[args[0]], mapping[args[1]]]).result
                case "scalar.string.length", [ir.StringType()]:
                    mapping[r] = db.RuntimeCall(mlirtypes.i64(), str_attr("StringLength"), [mapping[args[0]]]).result
                case "scalar.string.find", [ir.StringType(), ir.StringType(), ir.IntType(), ir.IntType()]:
                    mapping[r] = db.RuntimeCall(mlirtypes.i64(), str_attr("PyStringFind"),
                                                [mapping[args[0]], mapping[args[1]], mapping[args[2]],
                                                 mapping[args[3]]]).result
                case "scalar.string.rfind", [ir.StringType(), ir.StringType(), ir.IntType(), ir.IntType()]:
                    mapping[r] = db.RuntimeCall(mlirtypes.i64(), str_attr("PyStringRFind"),
                                                [mapping[args[0]], mapping[args[1]], mapping[args[2]],
                                                 mapping[args[3]]]).result
                case "scalar.string.substr", [ir.StringType(), ir.IntType(), ir.IntType()]:
                    length = arith.SubIOp(mapping[args[2]],mapping[args[1]]).result
                    offsetP1 = arith.AddIOp(mapping[args[1]], arith.ConstantOp(mlirtypes.i64(), 1).result).result
                    mapping[r] = db.RuntimeCall(db.StringType.get(curr_context), str_attr("Substring"),[mapping[args[0]], offsetP1, length]).result
                case "scalar.string.replace", [ir.StringType(), ir.StringType(), ir.StringType()]:
                    mapping[r] = db.RuntimeCall(db.StringType.get(curr_context), str_attr("Replace"),
                                                [mapping[args[0]], mapping[args[1]], mapping[args[2]]]).result
                case "scalar.int.from_string", [ir.StringType()]:
                    mapping[r] = db.CastOp(to_mlir_type(r.type), mapping[args[0]]).result
                case "scalar.string.concatenate", [ir.StringType(), ir.StringType()]:
                    mapping[r] = db.RuntimeCall(db.StringType.get(curr_context), str_attr("Concatenate"),
                                                [mapping[args[0]], mapping[args[1]]]).result
                case "scalar.string.at", [ir.StringType(), ir.IntType()]:
                    mapping[r] = db.RuntimeCall(db.StringType.get(curr_context), str_attr("Substring"),
                                                [mapping[args[0]],
                                                 arith.AddIOp(mapping[args[1]],
                                                              arith.ConstantOp(mlirtypes.i64(), 1).result).result,
                                                 arith.ConstantOp(mlirtypes.i64(), 1).result]).result

                case "scalar.string.iter", [ir.FunctionRefType(), ir.RecordType(), ir.RecordType(), ir.StringType()]:
                    str_length = db.RuntimeCall(mlirtypes.i64(), str_attr("StringLength"), [mapping[args[3]]]).result
                    str_length = arith.IndexCastOp(mlirtypes.index(), str_length).result
                    const1 = arith.ConstantOp(mlirtypes.index(), 1).result
                    str_lengthp1 = arith.AddIOp(str_length, const1).result
                    forOp = scf.ForOp(const1, str_lengthp1, const1, [mapping[args[2]]])
                    with mlir.InsertionPoint(forOp.body):
                        char = db.RuntimeCall(db.StringType.get(curr_context), str_attr("Substring"),
                                              [mapping[args[3]], forOp.induction_variable,const1]).result
                        next_iter_val = call(args[0], [mapping[args[1]], forOp.inner_iter_args[0], char], mapping)
                        scf.YieldOp([next_iter_val])
                    mapping[r] = forOp.result
                case "list.iter", [ir.FunctionRefType(), ir.RecordType(), ir.RecordType(), ir.ListType()]:
                    list_length = db.ListLengthOp(mapping[args[3]]).result
                    const0 = arith.ConstantOp(mlirtypes.index(), 0).result
                    const1 = arith.ConstantOp(mlirtypes.index(), 1).result
                    forOp = scf.ForOp(const0, list_length, const1, [mapping[args[2]]])
                    with mlir.InsertionPoint(forOp.body):
                        elem = db.ListGetOp(to_mlir_type(args[3].type.element_type), mapping[args[3]],
                                            forOp.induction_variable).result
                        next_iter_val = call(args[0], [mapping[args[1]], forOp.inner_iter_args[0], elem], mapping)
                        scf.YieldOp([next_iter_val])
                    mapping[r] = forOp.result

                case "list.create", []:
                    mapping[r] = db.CreateListOp(to_mlir_type(r.type)).result
                case "list.at", [ir.ListType(), ir.IntType()]:
                    as_index = arith.IndexCastOp(mlirtypes.index(), mapping[args[1]]).result
                    mapping[r] = db.ListGetOp(to_mlir_type(r.type), mapping[args[0]], as_index).result
                case "list.append", [ir.ListType(), elem_type]:
                    db.ListAppendOp(mapping[args[0]], mapping[args[1]])
                case "scalar.string.split", [ir.StringType(), ir.StringType(), ir.IntType()]:
                    mapping[r] = db.RuntimeCall(db.ListType.get(db.StringType.get(curr_context)),
                                                str_attr("StringSplit"),
                                                [mapping[args[0]], mapping[args[1]], mapping[args[2]]]).result
                case "scalar.string.ord", [ir.StringType()]:
                    mapping[r] = db.RuntimeCall(mlirtypes.i64(), str_attr("Ord"), [mapping[args[0]]]).result
                case "range.iter", [ir.FunctionRefType(), ir.RecordType(), ir.RecordType(), ir.IntType(), ir.IntType(), ir.IntType()]:
                    start = arith.IndexCastOp(mlirtypes.index(), mapping[args[3]]).result
                    end = arith.IndexCastOp(mlirtypes.index(), mapping[args[4]]).result
                    step = arith.IndexCastOp(mlirtypes.index(), mapping[args[5]]).result
                    negativeStep = arith.CmpIOp(arith.CmpIPredicate.slt, step, arith.ConstantOp(mlirtypes.index(), 0).result).result
                    whileOp = scf.WhileOp([mlirtypes.index(),mapping[args[2]].type], [start, mapping[args[2]]])
                    beforeBlock=whileOp.before.blocks.append()
                    afterBlock=whileOp.after.blocks.append()
                    uLoc = mlir.Location.unknown()
                    iArgBefore = beforeBlock.add_argument(mlirtypes.index(),uLoc)
                    iArgAfter = afterBlock.add_argument(mlirtypes.index(),uLoc)
                    iterArgBefore = beforeBlock.add_argument(mapping[args[2]].type,uLoc)
                    iterArgAfter = afterBlock.add_argument(mapping[args[2]].type,uLoc)
                    with mlir.InsertionPoint(beforeBlock):
                        # if step<0 ? itervar[0]> end : itervar[0] < end
                        cond1 = arith.CmpIOp(arith.CmpIPredicate.sgt, iArgBefore, end).result
                        cond2 = arith.CmpIOp(arith.CmpIPredicate.slt, iArgBefore, end).result
                        cond = arith.SelectOp(negativeStep, cond1, cond2).result
                        scf.ConditionOp(cond, [iArgBefore, iterArgBefore])
                    with mlir.InsertionPoint(afterBlock):
                        iAsI64 = arith.IndexCastOp(mlirtypes.i64(), iArgAfter).result
                        next_iter_val = call(args[0], [mapping[args[1]],  iterArgAfter, iAsI64], mapping)
                        next_itervar = arith.AddIOp(iArgAfter, step).result
                        scf.YieldOp([next_itervar, next_iter_val])
                    mapping[r] = whileOp.results[1]

                case "dbg.print", [ir.StringType()]:
                    db.RuntimeCall(None, str_attr("DumpValue"), [mapping[args[0]]])
                case "list.set", [ir.ListType(), ir.IntType(), elem_type]:
                    as_index = arith.IndexCastOp(mlirtypes.index(), mapping[args[1]]).result
                    db.ListSetOp(mapping[args[0]], as_index, mapping[args[2]])
                case "scalar.int.to_string", [ir.IntType()]:
                    mapping[r] = db.CastOp(to_mlir_type(r.type), mapping[args[0]]).result
                case _:
                    print("Can not translate op", name, "for types", arg_types, file=sys.stderr)
                    mapping[r] = util.UndefOp(to_mlir_type(r.type)).result
        case ir.Call(result=r, name=callee, args=args):
            mlir_args = [mapping[arg] for arg in args]
            if (isinstance(r.type, ir.VoidType)):
                func.CallOp([], callee, mlir_args)
                # c = dbpy.ConstantOp(to_mlir_type(r.type), str_attr("None"))
                assert False
            else:
                c = func.CallOp([to_mlir_type(r.type)], callee, mlir_args)
            mapping[r] = c.result
        case ir.IfElse(cond=cond, ifBody=ifBody, elseBody=elseBody, return_types=return_types):
            if_stmt = stmt
            if_op = scf.IfOp(mapping[cond], [to_mlir_type(t) for t in return_types], hasElse=True)
            with mlir.InsertionPoint(if_op.then_block):
                for stmt in ifBody.ops:
                    to_mlir_stmt(stmt, mapping)
            with mlir.InsertionPoint(if_op.else_block):
                for stmt in elseBody.ops:
                    to_mlir_stmt(stmt, mapping)
            for r1, r2 in zip(if_op.results, if_stmt.results):
                mapping[r2] = r1
        case ir.Yield(values=values):
            scf.YieldOp([mapping[value] for value in values])
        case ir.FunctionRef(result=r, name=name):
            mapping[r] = func.ConstantOp(to_mlir_type(r.type), mlir.FlatSymbolRefAttr.get(name)).result
        case ir.MakeRecord(result=r, res_type=res_type, values=values):
            ordered_members = [m for m, t in res_type.members]
            if len(ordered_members)==0:
                mapping[r] = util.UndefOp(to_mlir_type(res_type)).result
            else:
                mapping[r] = util.PackOp(to_mlir_type(res_type), [mapping[values[m]] for m in ordered_members]).result
        case ir.RecordGet(result=r, record=record, member=member):
            match record.type:
                case ir.RecordType(members=members):
                    for m, t in members:
                        if m == member:
                            member_index = members.index((m, t))
                            break
                    else:
                        assert False, f"Member {member} not found in record {record.type}"
                    index_attr = mlir.IntegerAttr.get(mlir.IntegerType.get_signless(32), member_index)
                    mapping[r] = util.GetTupleOp(to_mlir_type(r.type), mapping[record], index_attr).result
                case _:
                    assert False, f"Expected record type, got {record.type}"
        case _:
            print(stmt)
            assert False


def to_mlir_func(fn):
    mapping = {}
    mlir_func = func.FuncOp(fn.name, ([to_mlir_type(t.type) for t in fn.args],
                                      [] if isinstance(fn.res_type, ir.VoidType) else [
                                          to_mlir_type(fn.res_type)]))  # todo: translate function signature
    with mlir.InsertionPoint(mlir_func.add_entry_block()), mlir.Location.unknown():
        for i, arg in enumerate(fn.args):
            mapping[arg] = mlir_func.body.blocks[0].arguments[i]
        had_return = False
        for stmt in fn.body.ops:
            if isinstance(stmt, ir.Return):
                had_return = True
            to_mlir_stmt(stmt, mapping)
        if not had_return:
            func.ReturnOp([])
        return mlir_func


def to_mlir_module(module):
    global curr_context
    global curr_module
    context = mlir.Context()
    curr_context = context  # todo: remove this hack
    mlir_init.init_context(context)
    with context, mlir.Location.unknown():
        mlir_module = mlir.Module.create()
        curr_module = mlir_module
        with mlir.InsertionPoint(mlir_module.body), mlir.Location.unknown():
            for func in module.funcs():
                to_mlir_func(func)
        return mlir_module


def compile(module):
    mlir_module = to_mlir_module(module)
    return mlir_module
