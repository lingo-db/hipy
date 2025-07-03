from typing import List, Tuple, Dict

import hipy
from hipy import ir
from hipy.lib.builtins import tuple as _tuple, _concrete_dict, _concrete_list, _const_str
from hipy.lib.numpy import ndarray
from hipy.value import Type, ValueHolder, CValue, HLCFunctionValue, Value, TypeValue
from hipy import intrinsics


@hipy.classdef
class row(Value):
    def __init__(self, column_types: Dict[str, Type], value):
        super().__init__(value)
        self._column_types = column_types

    class RowType(Type):
        def __init__(self, columns: Dict[str, Type]):
            assert columns is not None
            self.columns = columns

        def ir_type(self):
            return ir.RecordType([(n, t.ir_type()) for n, t in self.columns.items()])

        def construct(self, value, context):
            return row(self.columns, value)

        def __eq__(self, other):
            if isinstance(other, row.RowType):
                return self.columns == other.columns
            else:
                return False

    def __hipy_get_type__(self):
        return row.RowType(self._column_types)

    @staticmethod
    def __hipy_create_type__(column_types) -> Type:
        return row.RowType(column_types)

    @hipy.raw
    def __getitem__(self, item, _context):
        match item:
            case ValueHolder(value=_const_str(cval=name)):
                res_type=self.value._column_types[name]
                return _context.wrap(res_type.construct(ir.RecordGet(_context.block,res_type.ir_type(),self.value.__value__, name).result, _context))
            case _:
                raise NotImplementedError()

    @hipy.compiled_function
    def __topython__(self):
        intrinsics.not_implemented()


@hipy.classdef
class table(Value):
    __HIPY_MUTABLE__ = False

    def __init__(self, table_type, column_types, value):
        super().__init__(value)
        self._table_type = table_type
        self._column_types = column_types

    @staticmethod
    @hipy.compiled_function
    def __create__(value):
        if intrinsics.isa(value, _concrete_dict):
            for k in value:
                if not intrinsics.isa(value[k], column):
                    intrinsics.not_implemented()
            record = intrinsics.create_record([k for k in value], [value[k] for k in value])
            return intrinsics.call_builtin("table.from_dict",
                                           intrinsics.create_type(table, [(k, value[k]._element_type) for k in value]),
                                           [record],side_effects=False)

        else:
            intrinsics.not_implemented()

    class TableType(Type):
        def __init__(self, columns: List[Tuple[str, Type]]):
            self.columns = columns

        def ir_type(self):
            return ir.TableType([(n, t.ir_type()) for n, t in self.columns])

        def construct(self, value, context):
            column_types = context.wrap(_concrete_dict({n: context.wrap(TypeValue(t)) for n, t in self.columns}))
            return table(table_type=self, column_types=column_types, value=value)

        def __eq__(self, other):
            if isinstance(other, table.TableType):
                return self.columns == other.columns
            else:
                return False

    def __hipy_get_type__(self):
        return self._table_type

    @staticmethod
    def __hipy_create_type__(column_types) -> Type:
        return table.TableType(column_types)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.call_builtin("table.to_python", object, [self])

    @hipy.compiled_function
    def __hipy__repr__(self):
        return repr(intrinsics.to_python(self))

    @hipy.compiled_function
    def __str__(self):
        return str(intrinsics.to_python(self))

    @hipy.compiled_function
    def sort(self, by, ascending):
        return intrinsics.call_builtin("table.sort", intrinsics.typeof(self), [self, [x for x in by], ascending])

    @hipy.compiled_function
    def filter_by_column(self, c):
        if intrinsics.isa(c, column):
            return intrinsics.call_builtin("table.filter", intrinsics.typeof(self), [self, c])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def get_column(self, name):
        if intrinsics.isa(name, str):
            return intrinsics.call_builtin("table.get_column", intrinsics.create_type(column, self._column_types[name]),
                                           [self],side_effects=False,attributes={"column":name})
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def select_columns(self, names):
        if intrinsics.isa(names, _concrete_list):
            for n in names:
                if not intrinsics.isa(n, str):
                    intrinsics.not_implemented()
            return intrinsics.call_builtin("table.select",
                                           intrinsics.create_type(table, [(n, self._column_types[n]) for n in names]),
                                           [self],side_effects=False,attributes={"columns":names})
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def get_slice(self, s):
        if intrinsics.isa(s, slice):
            if s.step is not None:
                intrinsics.not_implemented()
            length = len(self)
            start = s.start if s.start is not None else 0
            stop = s.stop if s.stop is not None else length
            start = start if start >= 0 else length + start
            stop = stop if stop >= 0 else length + stop
            return intrinsics.call_builtin("table.slice", intrinsics.typeof(self), [self, start, stop])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __len__(self):
        return intrinsics.call_builtin("table.length", int, [self],side_effects=False)

    @hipy.compiled_function
    def set_column(self, name, c):
        if intrinsics.isa(name, _const_str) and intrinsics.isa(c, column):
            new_columns =  []
            added=False
            for n in self._column_types:
                if n == name:
                    added=True
                    new_columns.append((name, c._element_type))
                else:
                    new_columns.append((n, self._column_types[n]))
            if not added:
                new_columns.append((name, c._element_type))
            return intrinsics.call_builtin("table.set_column", intrinsics.create_type(table, new_columns),
                                           [self, c], side_effects=False, attributes={"column":name})
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def apply_row_wise(self, fn):
        row_type = intrinsics.create_type(row, self._column_types)
        bound_fn = intrinsics.bind(fn, [row_type])
        return intrinsics.call_builtin("table.apply_row_wise_scalar", intrinsics.create_type(column, bound_fn.res_type),
                                       [self, bound_fn],side_effects=False)


    @hipy.compiled_function
    def join_inner(self, other, left_on, right_on,left_keep,right_keep):
        if intrinsics.isa(other, table):
            new_columns =  []
            for l in left_keep:
                new_columns.append((l, self._column_types[l]))
            for r in right_keep:
                new_columns.append((r, other._column_types[r]))

            return intrinsics.call_builtin("table.join_inner", intrinsics.create_type(table, new_columns), [self, other],attributes={"left_on":left_on,"right_on":right_on, "left_keep":left_keep, "right_keep":right_keep},side_effects=False)
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def join_left(self, other, left_on, right_on,left_keep,right_keep):
        if intrinsics.isa(other, table):
            new_columns =  []
            for l in left_keep:
                new_columns.append((l, self._column_types[l]))
            for r in right_keep:
                new_columns.append((r, other._column_types[r]))

            return intrinsics.call_builtin("table.join_left", intrinsics.create_type(table, new_columns), [self, other],attributes={"left_on":left_on,"right_on":right_on, "left_keep":left_keep, "right_keep":right_keep},side_effects=False)
        else:
            intrinsics.not_implemented()
    @staticmethod
    @hipy.raw
    def _tpl_get_item_types(tupletype,_context):
        match tupletype:
            case ValueHolder(value=TypeValue(type=hipy.lib.builtins.tuple.TupleType(element_types=element_types))):
                return _context.create_list([_context.wrap(TypeValue(t)) for t in element_types])
            case _:
                raise NotImplementedError()
    @hipy.compiled_function
    def aggregate(self, group_by, aggregations):
        input_cols=[a[1] for a in aggregations]
        output_cols=[a[0] for a in aggregations]
        input_col_types=[self._column_types[c] for c in input_cols]
        agg_types=[intrinsics.typeof(a[2]) for a in aggregations]
        c_aggs=intrinsics.as_constant(aggregations)
        def agg_fn(agg,row):
            res = []
            i=0
            for a in c_aggs:
                res.append(a[3](agg[i],row[i]))
                i+=1
            return tuple._from_const_list(res)
        def finalize_fn(agg):
            res = []
            i=0
            for a in c_aggs:
                res.append(a[4](agg[i]))
                i+=1
            return tuple._from_const_list(res)

        def init_fn():
            res = []
            for a in c_aggs:
                res.append(a[2])
            return tuple._from_const_list(res)

        agg_tuple_type=intrinsics.create_type(tuple, agg_types)
        row_tuple_type=intrinsics.create_type(tuple, input_col_types)
        bound_agg_fn = intrinsics.bind(agg_fn, [agg_tuple_type, row_tuple_type])
        bound_finalize_fn = intrinsics.bind(finalize_fn, [agg_tuple_type])
        bound_init_fn = intrinsics.bind(init_fn, [])
        new_columns = []
        for c in group_by:
            new_columns.append((c, self._column_types[c]))
        output_types=table._tpl_get_item_types(bound_finalize_fn.res_type)
        for i in range(0,len(output_cols)):
            new_columns.append((output_cols[i], output_types[i]))

        return intrinsics.call_builtin("table.aggregate", intrinsics.create_type(table, new_columns), [self,bound_init_fn,bound_agg_fn,bound_finalize_fn],attributes={"group_by":group_by,"input":input_cols,"output":output_cols},side_effects=False)

@hipy.classdef
class column(Value):
    __HIPY_MUTABLE__ = False

    def __init__(self, element_type, value):
        super().__init__(value)
        self._element_type = element_type

    @staticmethod
    @hipy.compiled_function
    def __create__(value):
        if intrinsics.isa(value, list):
            if value._element_type==object:
                intrinsics.not_implemented()
            return intrinsics.call_builtin("column.from_list", intrinsics.create_type(column, value._element_type),
                                           [value],side_effects=False)
        elif intrinsics.isa(value, ndarray):
            return intrinsics.call_builtin("column.from_array", intrinsics.create_type(column, value._dtype),
                                           [value],side_effects=False)
        else:
            intrinsics.not_implemented()

    @staticmethod
    @hipy.compiled_function
    def sequential(length):
        return intrinsics.call_builtin("column.sequential", intrinsics.create_type(column, int), [length],side_effects=False)

    class ColumnType(Type):
        def __init__(self, element_type: Type):
            self.element_type = element_type

        def ir_type(self):
            return ir.ColumnType(self.element_type.ir_type())

        def construct(self, value, context):
            return column(self.element_type, value)

        def __eq__(self, other):
            if isinstance(other, column.ColumnType):
                return self.element_type == other.element_type
            else:
                return False

    def __hipy_get_type__(self):
        return column.ColumnType(self._element_type)

    @staticmethod
    def __hipy_create_type__(element_type) -> Type:
        return column.ColumnType(element_type)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.call_builtin("column.to_python", object, [self])

    @hipy.compiled_function
    def __hipy__repr__(self):
        # this avoids going through the context -> no sideeffects
        return repr(intrinsics.call_builtin("column.to_python", object, [self]))

    @hipy.compiled_function
    def __str__(self):
        # this avoids going through the context -> no sideeffects
        return str(intrinsics.call_builtin("column.to_python", object, [self]))

    @hipy.compiled_function
    def apply(self, fn):
        bound_fn = intrinsics.bind(fn, [self._element_type])
        return intrinsics.call_builtin("column.apply_scalar", intrinsics.create_type(column, bound_fn.res_type),
                                       [self, bound_fn], side_effects=False)

    @hipy.compiled_function
    def element_wise(self, other, fn):
        if intrinsics.isa(other, column):
            bound_fn = intrinsics.bind(fn, [self._element_type, other._element_type])
            return intrinsics.call_builtin("column.binary_op", intrinsics.create_type(column, bound_fn.res_type),
                                           [self, other, bound_fn],side_effects=False)
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def aggregate(self, initial, fn, combine):
        initial_type = intrinsics.typeof(initial)
        bound_fn = intrinsics.bind(fn, [initial_type, self._element_type])
        bound_combine = intrinsics.bind(combine, [initial_type, initial_type])
        return intrinsics.call_builtin("column.aggregate", bound_combine.res_type,
                                       [self, initial, bound_fn, bound_combine])

    @hipy.compiled_function
    def filter_by_column(self, c):
        return intrinsics.call_builtin("column.filter", intrinsics.create_type(column, self._element_type), [self, c])

    @hipy.compiled_function
    def unique(self):
        return intrinsics.call_builtin("column.unique", intrinsics.create_type(column, self._element_type), [self])

    @hipy.compiled_function
    def isin(self, other):
        if intrinsics.isa(other, column):
            return intrinsics.call_builtin("column.isin_column", intrinsics.create_type(column, bool), [self, other])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __len__(self):
        return intrinsics.call_builtin("column.length", int, [self],side_effects=False)

    @hipy.compiled_function
    def get_by_index(self,pos):
        return intrinsics.call_builtin("column.value_by_index",  self._element_type, [self,pos])