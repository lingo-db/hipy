import abc
import textwrap
from typing import Dict, List, Any, Tuple

from hipy.cppbackend import cppir
from hipy.opt.pattern_rewriter import PatternRewriter, RewritePattern
import hipy.ir as ir


def get_str_constant(value):
    match value.producer:
        case ir.Constant(v=value):
            return value
    return None


class Operator(abc.ABC):

    @staticmethod
    def call_fn(fn, args, block: ir.Block):
        match fn.producer:
            case ir.FunctionRef(name=fn_name, closure=closure):
                return ir.Call(block, fn_name, (args + [closure]) if closure else args, fn.type.res_type).result

    @abc.abstractmethod
    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: "Operator"):
        pass

    @abc.abstractmethod
    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        pass


class TableScan(Operator):
    def __init__(self, table: ir.SSAValue):
        self.table = table

    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: Operator):
        iterate_table = cppir.IterateTable(block, self.table, required_cols)
        parent.consume({c:v for c,v in zip(iterate_table.required_cols,iterate_table.iter_vars)}, iterate_table.iter_block, module)

    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        pass

    def __str__(self):
        return f"TableScan({self.table})"

class InnerJoin(Operator):
    def __init__(self, left: Operator, right: Operator, left_key, right_key, left_keep: bool, right_keep: bool, left_types, right_types):
        self.left = left
        self.right = right
        self.left_key = left_key
        self.right_key = right_key
        self.left_keep = left_keep
        self.right_keep = right_keep
        self.left_types = left_types
        self.right_types = right_types


    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: Operator):
        self.parent = parent
        self.left_output_cols=[c for c in required_cols if c in self.left_keep]
        self.right_output_cols=[c for c in required_cols if c in self.right_keep]
        self.join_ht_type=cppir.JoinHtType([self.right_types[c] for c in self.right_key], [self.right_types[c] for c in self.right_output_cols])
        self.join_ht=cppir.CreateJoinHt(block, self.join_ht_type).result
        self.phase="build"
        self.right.produce(list(set(self.right_output_cols+self.right_key)), block, module, self)
        cppir.JoinHtBuild(block, self.join_ht)
        self.phase="probe"
        self.left.produce(list(set(self.left_output_cols+self.left_key)), block, module, self)

    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        if self.phase=="build":
            cppir.JoinHtInsert(block, self.join_ht, [cols[c] for c in self.right_key], [cols[c] for c in self.right_output_cols])
        else:
            lookup=cppir.JoinHtLookup(block, self.join_ht, [cols[c] for c in self.left_key])
            self.parent.consume({**cols, **{c:v for c,v in zip(self.right_output_cols,lookup.iter_vars)}}, lookup.iter_block, module)

    def __str__(self):
        return f"InnerJoin({self.left_key},{self.right_key})\n{textwrap.indent(str(self.left), '  ')}\n{textwrap.indent(str(self.right), '  ')}"

class LeftJoin(Operator):
    def __init__(self, left: Operator, right: Operator, left_key, right_key, left_keep: bool, right_keep: bool, left_types, right_types):
        self.left = left
        self.right = right
        self.left_key = left_key
        self.right_key = right_key
        self.left_keep = left_keep
        self.right_keep = right_keep
        self.left_types = left_types
        self.right_types = right_types


    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: Operator):
        self.parent = parent
        self.left_output_cols=[c for c in required_cols if c in self.left_keep]
        self.right_output_cols=[c for c in required_cols if c in self.right_keep]
        self.join_ht_type=cppir.JoinHtType([self.right_types[c] for c in self.right_key], [self.right_types[c] for c in self.right_output_cols])
        self.join_ht=cppir.CreateJoinHt(block, self.join_ht_type).result
        self.phase="build"
        self.right.produce(list(set(self.right_output_cols+self.right_key)), block, module, self)
        cppir.JoinHtBuild(block, self.join_ht)
        self.phase="probe"
        self.left.produce(list(set(self.left_output_cols+self.left_key)), block, module, self)

    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        if self.phase=="build":
            cppir.JoinHtInsert(block, self.join_ht, [cols[c] for c in self.right_key], [cols[c] for c in self.right_output_cols])
        else:
            flag=cppir.CreateFlag(block, cppir.FlagType()).result

            lookup=cppir.JoinHtLookup(block, self.join_ht, [cols[c] for c in self.left_key])
            cppir.SetFlag(lookup.iter_block, flag)
            self.parent.consume({**cols, **{c:v for c,v in zip(self.right_output_cols,lookup.iter_vars)}}, lookup.iter_block, module)

            ifop=ir.IfElse(block, cppir.CheckFlag(block, flag).result, [])
            self.parent.consume({**cols, **{c:ir.Constant(ifop.elseBody, float("NAN"), self.right_types[c]).result for c in self.right_output_cols}}, ifop.elseBody, module)
            ir.Yield(ifop.ifBody, [])
            ir.Yield(ifop.elseBody, [])

    def __str__(self):
        return f"InnerJoin({self.left_key},{self.right_key})\n{textwrap.indent(str(self.left), '  ')}\n{textwrap.indent(str(self.right), '  ')}"
class Aggregation(Operator):
    def __init__(self, group_by: List[str], input: List[str], output: List[str], child: Operator, init_fn: ir.SSAValue, agg_fn: ir.SSAValue, finalize_fn: ir.SSAValue,types:Dict[str,Any],res_types):
        self.group_by = group_by
        self.input = input
        self.output = output
        self.child = child
        self.init_fn = init_fn
        self.agg_fn = agg_fn
        self.finalize_fn = finalize_fn
        self.key_types = [types[c] for c in group_by]
        self.res_types = [res_types[c] for c in output]

    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: Operator):
        self.parent = parent
        self.agg_ht_type=cppir.AggregationHtType(self.key_types, self.init_fn.type.res_type)
        self.agg_ht=cppir.CreateAggregationHt(block, self.agg_ht_type).result
        self.child.produce(list(set(required_cols+self.group_by+self.input)), block, module, self)
        iterate_agght=cppir.IterateAggregationHt(block,self.agg_ht)
        keys={c:iterate_agght.iter_key_vars[i] for i,c in enumerate(self.group_by)}
        final_val=Operator.call_fn(self.finalize_fn, [iterate_agght.iter_val], iterate_agght.iter_block)
        agg_values={c:ir.RecordGet(iterate_agght.iter_block, t, final_val, f"_elt{i}").result for i, (t,c) in enumerate(zip(self.res_types, self.output))}
        self.parent.consume({**keys, **agg_values}, iterate_agght.iter_block, module)

    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        key=[cols[c] for c in self.group_by]
        input=[cols[c] for c in self.input]
        agg=cppir.Aggregate(block, self.agg_ht, key)
        initial_value=Operator.call_fn(self.init_fn, [],agg.init_block)
        ir.Yield(agg.init_block, [initial_value])
        curr_row=ir.MakeRecord(agg.agg_block, ir.RecordType([(f"_elt{i}",v.type) for i, v in enumerate(input)]) ,{f"_elt{i}": v for i,v in enumerate(input)}).result
        agg_res=Operator.call_fn(self.agg_fn, [agg.agg_val, curr_row], agg.agg_block)
        ir.Yield(agg.agg_block, [agg_res])


    def __str__(self):
        return f"Aggregation()\n{textwrap.indent(str(self.child), '  ')}"
class Project(Operator):
    def __init__(self, cols: List[str], child: Operator):
        self.cols = cols
        self.child = child

    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: Operator):
        self.parent = parent
        self.child.produce(required_cols, block, module, self)

    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        self.parent.consume(cols, block, module)

    def __str__(self):
        return f"Project({self.cols})\n{textwrap.indent(str(self.child), '  ')}"


class AddIndexColumn(Operator):
    def __init__(self, col, child: Operator):
        self.col = col
        self.child = child

    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: Operator):
        self.parent = parent
        self.counter=cppir.CreateCounter(block, cppir.CounterType()).result
        self.child.produce([c for c in required_cols if c!=self.col], block, module, self)

    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        v=cppir.IncrementCounter(block, self.counter).result
        self.parent.consume({**cols,self.col:v}, block, module)

    def __str__(self):
        return f"AddIndexColumn({self.col})\n{textwrap.indent(str(self.child), '  ')}"

class Filter(Operator):
    def __init__(self, required_cols: List[str], child: Operator, filter_fn: ir.SSAValue):
        self.filter_fn = filter_fn
        self.child = child
        self.required_cols = required_cols

    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: Operator):
        self.parent=parent
        required_cols = set(required_cols)
        required_cols.update(self.required_cols)
        self.child.produce(list(required_cols), block, module, self)

    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        required_cols = [cols[c] for c in self.required_cols]
        output = Operator.call_fn(self.filter_fn, required_cols, block)
        ifop=ir.IfElse(block, output, [])
        self.parent.consume(cols, ifop.ifBody, module)
        ir.Yield(ifop.ifBody,[])

        ir.Yield(ifop.elseBody,[])

    def __str__(self):
        return f"Filter({self.required_cols})\n{textwrap.indent(str(self.child), '  ')}"


class Map(Operator):
    def __init__(self, required_cols: List[str], output_cols: List[str], child: Operator, map_fn: ir.SSAValue):
        self.map_fn = map_fn
        self.child = child
        self.required_cols = required_cols
        self.output_cols = output_cols

    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: Operator):
        self.parent = parent
        required_cols = set(required_cols)
        required_cols = required_cols.difference(self.output_cols)
        required_cols.update(self.required_cols)
        self.child.produce(list(required_cols), block, module, self)

    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        required_cols = [cols[c] for c in self.required_cols]
        output = Operator.call_fn(self.map_fn, required_cols, block)
        if len(self.output_cols) == 1:
            cols[self.output_cols[0]] = output
        else:
            col_types= {c: t for c, t in output.type.members}
            for c in self.output_cols:
                cols[c] = ir.RecordGet(block, col_types[c], output, ir.Constant(block,c,ir.string).result).result

        self.parent.consume(cols, block, module)


    def __str__(self):
        return f"Map({self.required_cols}->{self.output_cols})\n{textwrap.indent(str(self.child), '  ')}"


class Materialize(Operator):
    def __init__(self, required_cols: List[Tuple[str, Any]], child: Operator):
        self.child = child
        self.required_cols = required_cols

    def produce(self, required_cols, block: ir.Block, module: ir.Module, parent: Operator):
        table_builder_type = cppir.TableBuilderType(self.required_cols)
        self.table_builder = cppir.CreateTableBuilder(block, table_builder_type).result
        self.child.produce([n for n, t in self.required_cols], block, module, self)
        self.result=cppir.TableBuilderFinish(block, self.table_builder, ir.TableType(self.required_cols)).result

    def consume(self, cols: Dict[str, ir.SSAValue], block: ir.Block, module: ir.Module):
        cppir.TableBuilderAppend(block,self.table_builder, {c:cols[c] for c,t in self.required_cols})

    def __str__(self):
        return f"Materialize({self.required_cols})\n{textwrap.indent(str(self.child), '  ')}"
def optimize(tree:Operator):
    #print("optimizing:",tree)
    match tree:
        case InnerJoin(left=left,right=right, left_key=left_key,left_keep=left_keep):
            match left:
                case Map(output_cols=computed,child=child):
                    if len(set(left_key).intersection(set(computed)))==0:
                        mapOp=left
                        tree.left=child
                        tree.left_keep= [k for k in left_keep if k not in computed]
                        mapOp.child=tree
                        return optimize(mapOp)
                    else:
                        return tree
                case Filter(child=child):
                    filterOp=left
                    tree.left=child
                    filterOp.child=tree
                    return optimize(filterOp)
                case _:
                    return tree
        case AddIndexColumn() | Materialize() | Filter() | Map():
            tree.child=optimize(tree.child)
            return tree
        case _:
            return tree



class RelRewriter(RewritePattern):

    def construct_tree(self, v: ir.SSAValue, rewriter, first=False, sort_irelevant=False):
        tree = None
        op = v.producer
        if not first and v in rewriter.uses and len(rewriter.uses[v])>1:
            tree = self.construct_tree(v, rewriter,True)
            if tree is None:
                return TableScan(v)
            block = rewriter.before(v.producer)
            module = rewriter.module
            #print("early materialization")
            #print("tree:", tree)
            tree.produce([], block, module, None)
            rewriter.replace_with_value(v.producer, tree.result)
            tree=TableScan(tree.result)

        else:
            match op:
                case ir.CallBuiltin(name="table.sort", args=[table, order, ascending]):
                    if sort_irelevant:
                        tree= self.construct_tree(table, rewriter, sort_irelevant=True)
                    else:
                        if first:
                            return None
                        tree = TableScan(v)
                case ir.CallBuiltin(name="table.compute", args=[table, fn], attributes={"input": required_cols,"output": computed_cols}):
                    tree = Map(required_cols, computed_cols, self.construct_tree(table,rewriter, sort_irelevant=sort_irelevant), fn)
                case ir.CallBuiltin(name="table.filter_by_func", args=[table,fn],attributes={"columns": required_cols}):
                    tree = Filter(required_cols, self.construct_tree(table,rewriter, sort_irelevant=sort_irelevant), fn)
                case ir.CallBuiltin(name="table.select", args=[table],attributes={"columns": required_cols}):
                    tree = Project(required_cols, self.construct_tree(table,rewriter, sort_irelevant=sort_irelevant))
                case ir.CallBuiltin(name="table.add_index_column", args=[table],attributes={"name":name}):
                    tree = AddIndexColumn(name, self.construct_tree(table,rewriter, sort_irelevant=sort_irelevant))
                case ir.CallBuiltin(name="table.join_inner", args=[left, right],attributes={"left_on":left_key,"right_on":right_key,"left_keep": left_keep,"right_keep":right_keep}):
                    tree = InnerJoin(self.construct_tree(left,rewriter, sort_irelevant=sort_irelevant), self.construct_tree(right,rewriter, sort_irelevant=sort_irelevant), left_key, right_key, left_keep, right_keep, {c:t for c,t in left.type.members}, {c:t for c,t in right.type.members})
    
                case ir.CallBuiltin(name="table.join_left", args=[left, right],attributes={"left_on":left_key,"right_on":right_key,"left_keep": left_keep,"right_keep":right_keep}):
                    tree = LeftJoin(self.construct_tree(left,rewriter, sort_irelevant=sort_irelevant), self.construct_tree(right,rewriter, sort_irelevant=sort_irelevant), left_key, right_key, left_keep, right_keep, {c:t for c,t in left.type.members}, {c:t for c,t in right.type.members})
                case ir.CallBuiltin(name="table.aggregate", args=[table,init_fn, agg_fn,finalize_fn],attributes={"group_by":group_by,"input":input,"output":output}):
                    tree = Aggregation(group_by, input, output, self.construct_tree(table,rewriter,sort_irelevant=True), init_fn, agg_fn, finalize_fn, {c:t for c,t in table.type.members},{c:t for c,t in op.result.type.members})
                case _:
                    if first:
                        return None
                    tree = TableScan(v)

        if first:
            required_cols = [(m, t) for m, t in v.type.members]
            tree = Materialize(required_cols, tree)
        return tree
            

    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        if not any([isinstance(a.type, ir.TableType) for a in op.get_used_values()]):
            return False
        match op:
            #case ir.CallBuiltin(name="table.")
            case ir.CallBuiltin(name="table.select"):
                return False
            case ir.CallBuiltin(name="table.compute"):
                return False
            case ir.CallBuiltin(name="table.filter_by_func"):
                return False
            case ir.CallBuiltin(name="table.add_index_column"):
                return False
            case ir.CallBuiltin(name="table.join_inner"):
                return False
            case ir.CallBuiltin(name="table.aggregate"):
                return False
            case ir.IfElse:
                return False
            case _:
                for a in op.get_used_values():
                    if isinstance(a.type, ir.TableType):
                        tree = self.construct_tree(a, rewriter,True)
                        if tree is None:
                            return tree
                        tree=optimize(tree)
                        block = rewriter.before(a.producer)
                        module = rewriter.module
                        #print("op:",op)
                        #print("tree:",tree)
                        tree.produce([], block, module, None)
                        rewriter.replace_with_value(a.producer,tree.result)
                        #print(block)

                return False


def rewrite(module: ir.Module):
    rewriter = PatternRewriter(
        [RelRewriter()], module)
    rewriter.rewrite()
    return module
