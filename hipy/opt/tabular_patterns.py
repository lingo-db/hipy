import functools

from hipy.opt.pattern_rewriter import PatternRewriter, RewritePattern
import hipy.ir as ir
from collections import deque


def get_str_constant(value):
    match value.producer:
        case ir.Constant(v=value):
            return value
    return None


def make_str_constant(rewriter, value):
    return rewriter.create(ir.Constant, v=value, t=ir.StringType()).result


class FuncManager:
    def __init__(self):
        self.closure_vals = {}
        self.func_names = {}

    def add(self, func):
        match func.producer:
            case ir.FunctionRef(name=func_name, closure=closure):
                if closure is not None:
                    self.closure_vals[func] = closure
                self.func_names[func] = func_name
                return True
            case _:
                return False

    def closure_type(self):
        if not self.closure_vals:
            return None
        closure_types = [(f"f{v.id}", closure.type) for v, closure in self.closure_vals.items()]
        return ir.RecordType(closure_types)

    def closure_val(self, rewriter):
        if not self.closure_vals:
            return None
        closure_vals = {f"f{v.id}": closure for v, closure in self.closure_vals.items()}
        return rewriter.create(ir.MakeRecord, self.closure_type(), closure_vals).result

    def call(self, block, func, args, new_func):
        #print("calling func manager", func.id)
        args = [*args]
        if func in self.closure_vals:
            closure_val = new_func.args[-1]
            args.append(ir.RecordGet(block, self.closure_vals[func].type, closure_val, f"f{func.id}").result)
        return ir.Call(block, self.func_names[func], args, func.type.res_type).result


class RewriteFromDict(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="table.from_dict", args=[dict]):
                match dict.producer:
                    case ir.MakeRecord(values=values):
                        columns = []
                        idx_columns=[]
                        common_table = None
                        for col_name,val in values.items():
                            match val.producer:
                                case ir.CallBuiltin(name="table.get_column", args=[table], attributes={"column": col_name}):
                                    if common_table is None:
                                        common_table = table
                                    elif common_table != table:
                                        return False
                                    columns.append(col_name)
                                case ir.CallBuiltin(name="column.sequential", args=[len]):
                                    match len.producer:
                                        case ir.CallBuiltin(name="table.length", args=[table]):
                                            if common_table is None:
                                                common_table = table
                                            elif common_table != table:
                                                return False
                                            idx_columns.append(col_name)
                                case _:
                                    return False
                        res_table=rewriter.create( ir.CallBuiltin, name="table.select", args=[common_table], ret_type=op.result.type, attributes={"columns": columns},side_effects=False).result #todo: compute res_type
                        for idx_col in idx_columns:
                            res_table=rewriter.create(ir.CallBuiltin, name="table.add_index_column", args=[res_table], ret_type=op.result.type,side_effects=False, attributes={"name": idx_col}).result #todo: compute res_type
                        rewriter.replace_with_value(op, res_table)
                        return True


class RewriteSetComputedColumn(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="table.set_column", args=[table, column], attributes={"column": column_name}):
                match column.producer:
                    case ir.CallBuiltin(name="column.apply_scalar", args=[column2, func]):
                        match column2.producer:
                            case ir.CallBuiltin(name="table.get_column", args=[table2], attributes={"column": column_name2}):
                                if table == table2:
                                    rewriter.replace_with(op, ir.CallBuiltin, name="table.compute",
                                                          args=[table, func],
                                                          ret_type=op.result.type, side_effects=False, attributes={"input":[column_name2], "output":[column_name]})
                                    return True
                    case ir.CallBuiltin(name="column.binary_op", args=[column1, column2, func]):
                        match [column1.producer, column2.producer]:
                            case [ir.CallBuiltin(name="table.get_column", args=[table1], attributes={"column": column_name1}),
                                  ir.CallBuiltin(name="table.get_column", args=[table2], attributes={"column": column_name2})]:
                                if table1 == table2 and table == table1:

                                    rewriter.replace_with(op, ir.CallBuiltin, name="table.compute", args=[table1,func],
                                                          ret_type=op.result.type, side_effects=False, attributes={"input":[column_name1, column_name2], "output":[column_name]})
                                    return True

        return False

class RewriteAddIndexColumn(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="table.set_column", args=[table, column], attributes={"column": column_name}):
                match column.producer:
                    case ir.CallBuiltin(name="column.sequential", args=[len]):
                        match len.producer:
                            case ir.CallBuiltin(name="table.length", args=[table2]):
                                if table == table2:
                                    rewriter.replace_with(op, ir.CallBuiltin, name="table.add_index_column",
                                                          args=[table],
                                                          ret_type=op.result.type, side_effects=False, attributes={"name": column_name})
                                    return True

            case ir.CallBuiltin(name="column.sequential", args=[len]):
                match len.producer:
                    case ir.CallBuiltin(name="column.length", args=[column]):
                        match column.producer:
                            case ir.CallBuiltin(name="table.get_column", args=[table], attributes={"column": column_name}):
                                rewriter.replace_with(op, ir.CallBuiltin, name="column.sequential", ret_type=op.result.type, args=[rewriter.create(ir.CallBuiltin, name="table.length", ret_type=ir.i64,args=[table],side_effects=False).result], side_effects=False)
                                return True
                        return True
        return False


class RewriteSetColumnRowApply(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="table.set_column", args=[table, column], attributes={"column": column_name}):
                match column.producer:
                    case ir.CallBuiltin(name="table.apply_row_wise_scalar", args=[table2, func]):
                        could_not_resolve = False
                        if table != table2:
                            match table2.producer:
                                case ir.CallBuiltin(name="table.select", args=[table3]):
                                    if table3 != table:
                                        could_not_resolve = True
                                case _:
                                    could_not_resolve = True
                            if could_not_resolve:
                                tmp_table=rewriter.create(ir.CallBuiltin, name="table.set_column", args=[table2, column],ret_type=ir.TableType(members=table2.type.members+[(column_name, column.type.element_type)]), attributes={"column": column_name}, side_effects=False).result
                                tmp_column=rewriter.create(ir.CallBuiltin, name="table.get_column", args=[tmp_table], ret_type=column.type, attributes={"column": column_name},side_effects=False).result
                                rewriter.replace_with(op, ir.CallBuiltin, name="table.set_column", args=[table, tmp_column], ret_type=op.result.type, attributes={"column": column_name}, side_effects=False)
                                return True

                        funcs = FuncManager()
                        if not funcs.add(func):
                            return False
                        required_columns = [col for col, _ in table2.type.members]
                        closure_type = funcs.closure_type()
                        global fused_cntr
                        new_func = ir.Function(rewriter.module, f"fused_{fused_cntr}",
                                               [t for _, t in table2.type.members] + (
                                                   [closure_type] if closure_type is not None else []),
                                               func.type.res_type)
                        fused_cntr += 1
                        col_mapping = {col: new_func.args[i] for i, col in enumerate(required_columns)}
                        record_type = ir.RecordType([(col, t) for col, t in table2.type.members])
                        record = ir.MakeRecord(new_func.body, record_type, col_mapping).result
                        res = funcs.call(new_func.body, func, [record], new_func)
                        ir.Return(new_func.body, [res])
                        func_ref = rewriter.create(ir.FunctionRef, new_func, funcs.closure_val(rewriter)).result
                        rewriter.replace_with(op, ir.CallBuiltin, name="table.compute",
                                              args=[table, func_ref], ret_type=op.result.type, side_effects=False, attributes={"input": required_columns, "output": [column_name]})
                        return True
        return False

class FuseColumnApply(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="column.apply_scalar", args=[column, func]):
                match column.producer:
                    case ir.CallBuiltin(name="column.apply_scalar", args=[column2, func2]):
                        funcs = FuncManager()
                        if not funcs.add(func):
                            return False
                        if not funcs.add(func2):
                            return False
                        closure_type = funcs.closure_type()
                        global fused_cntr
                        new_func = ir.Function(rewriter.module, f"fused_{fused_cntr}",
                                               [column2.type.element_type] + (
                                                   [closure_type] if closure_type is not None else []),
                                               func.type.res_type)
                        fused_cntr += 1
                        res2 = funcs.call(new_func.body, func2, [new_func.args[0]], new_func)
                        res = funcs.call(new_func.body, func, [res2], new_func)
                        ir.Return(new_func.body, [res])
                        func_ref = rewriter.create(ir.FunctionRef, new_func, funcs.closure_val(rewriter)).result
                        rewriter.replace_with(op, ir.CallBuiltin, name="column.apply_scalar", args=[column2, func_ref], ret_type=op.result.type, side_effects=False)
                        return True
        return False



class RewriteFilter(RewritePattern):
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        match op:
            case ir.CallBuiltin(name="table.filter", args=[table, filter_column]):
                computations = {}
                column_positions = {}
                column_mapping = {}
                queue = deque()
                queue.append(filter_column.producer)
                funcs = FuncManager()
                while queue:
                    curr = queue.popleft()
                    match curr:
                        case ir.CallBuiltin(name="table.get_column", args=[curr_table], attributes={"column": column_name}):
                            if curr_table != table:
                                return False
                            column_pos = column_positions.setdefault(column_name, len(column_positions))
                            column_mapping[curr.result] = column_pos
                        case ir.CallBuiltin(name="column.apply_scalar", args=[column, func_val]):
                            if not funcs.add(func_val):
                                return False

                            def gen(func_val, column, block, computed):
                                return funcs.call(block, func_val, [computed[column]], new_func)

                            computations[curr.result] = functools.partial(gen, func_val, column)
                            queue.append(column.producer)
                        case ir.CallBuiltin(name="column.binary_op", args=[column1, column2, func_val]):
                            if not funcs.add(func_val):
                                return False

                            def gen(func_val, column1, column2, block, computed):
                                return funcs.call(block, func_val, [computed[column1], computed[column2]], new_func)

                            computations[curr.result] = functools.partial(gen, func_val, column1, column2)
                            queue.append(column1.producer)
                            queue.append(column2.producer)
                        case _:
                            return False

                func_types = [None] * len(column_positions)
                required_cols = [None] * len(column_positions)
                for col, pos in column_positions.items():
                    func_types[pos] = table.type.col_type(col)
                    required_cols[pos] = col
                closure_type = funcs.closure_type()
                global fused_cntr
                new_func = ir.Function(rewriter.module, f"fused_filter_{fused_cntr}",
                                       func_types + ([closure_type] if closure_type is not None else []),
                                       ir.bool)
                fused_cntr += 1
                mapping = {}
                for val, pos in column_mapping.items():
                    mapping[val] = new_func.args[pos]
                for val, func in reversed(computations.items()):
                    r = func(new_func.body, mapping)
                    mapping[val] = r
                ir.Return(new_func.body, [mapping[filter_column]])
                func_ref = rewriter.create(ir.FunctionRef, new_func, funcs.closure_val(rewriter)).result

                rewriter.replace_with(op, ir.CallBuiltin, name="table.filter_by_func",
                                      args=[table, func_ref],
                                      ret_type=op.result.type, side_effects=False,attributes={"columns": required_cols})

                return True
        return False


fused_cntr = 0






def rewrite_set_column(module: ir.Module):
    rewriter = PatternRewriter(
        [RewriteSetComputedColumn(), RewriteFilter(), RewriteSetColumnRowApply(),RewriteAddIndexColumn(),RewriteFromDict(),FuseColumnApply()], module)
    rewriter.rewrite()
    return module
