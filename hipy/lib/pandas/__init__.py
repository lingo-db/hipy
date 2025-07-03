__HIPY_MODULE__ = "pandas"

import json
from typing import List, Dict

import numpy as np
import pandas as _pd
import sys

import hipy
from hipy import intrinsics, ir
from hipy.lib._tabular import column, table
from hipy.lib.builtins import _concrete_dict, _const_str, _concrete_list
from hipy.value import SimpleType, Value, Type, raw_module, ValueHolder, TypeValue, CValue, VoidValue, static_object
import hipy.lib.numpy

hipy.register(sys.modules[__name__])

pd = raw_module(_pd)
raw_np = raw_module(np)


class _NotRelevantType(Type):
    def ir_type(self):
        raise NotImplementedError()

    def construct(self, value, context):
        raise NotImplementedError()

    def __eq__(self, other):
        return False

    def __repr__(self):
        return "NotRelevantType"


@hipy.compiled_function
def _to_native_type(x):
    if intrinsics.isa(x, int):
        return np.int64(x)
    elif intrinsics.isa(x, float):
        return np.float64(x)
    else:
        return x


@hipy.compiled_function
def _to_python_type(x):
    if intrinsics.isa(x, np.int64):
        return int(x)
    elif intrinsics.isa(x, np.int64):
        return float(x)
    else:
        return x

@hipy.classdef
class Index(Value):
    __HIPY_MUTABLE__ = False

    def __init__(self, column, name, dtype, concrete_values=None):
        super().__init__(None)
        self._column = column
        self.name = name
        self.dtype = dtype
        self._concrete_values = concrete_values

    @staticmethod
    @hipy.compiled_function
    def __create__(data=None, name=None, dtype=None):
        if data is None:
            intrinsics.not_implemented()
        raw_data = None
        if intrinsics.isa(data, _concrete_list):
            all_const = True
            for val in data:
                if not intrinsics.isa(val, CValue):
                    all_const = False
            if all_const:
                raw_data = [val for val in data]
        c = column(data)
        return Index._create_raw(c, name, dtype, raw_data)

    def __abstract__(self, context):
        return ValueHolder.AbstractViaPython()

    @staticmethod
    @hipy.raw
    def _create_raw(column, name, dtype, raw_data=None, _context=None):
        concrete_values = None
        match raw_data:
            case ValueHolder(value=_concrete_list(items=items)):
                concrete_values = []
                for item in items:
                    if isinstance(item.value,CValue):
                        concrete_values.append(item.value.cval)
        return _context.wrap(Index(column, name, dtype, concrete_values=concrete_values))

    @hipy.compiled_function
    def __topython__(self):
        return pd.Index(self._column, name=self.name, dtype=self.dtype)

    @hipy.compiled_function
    def __hipy__repr__(self):
        repr(intrinsics.to_python(self))

    @hipy.compiled_function
    def __str__(self):
        return str(intrinsics.to_python(self))

    @staticmethod
    def __hipy_create_type__() -> Type:
        return _NotRelevantType()

    def __hipy_get_type__(self) -> Type:
        return _NotRelevantType()

    @hipy.compiled_function
    def _update(self, table):
        return Index._create_raw(table.get_column(self._colname()), self.name, self.dtype)

    @hipy.compiled_function
    def _colname(self):
        return "index" if self.name is None else ("index" + self.name)

    @hipy.compiled_function
    def _columns(self):
        return {self._colname(): self._column}

    @hipy.raw
    def _const_lookup(self, key, _context):
        if self.value._concrete_values is not None:
            match key:
                case ValueHolder(value=CValue(cval=cval)):
                    try:
                        return _context.constant(self.value._concrete_values.index(cval))
                    except ValueError:
                        pass
        return _context.constant(None)

    @hipy.compiled_function
    def _lookup(self, key):
        const_lookup = self._const_lookup(key)
        if const_lookup is None:
            return self._column.apply(lambda x: x == key)
        else:
            return const_lookup

@hipy.classdef
class MultiIndex(Value):
    __HIPY_MUTABLE__ = False

    def __init__(self, columns, names, dtype):
        super().__init__(None)
        self._cols = columns
        self.names = names
        self.dtype = dtype
    @staticmethod
    @hipy.compiled_function
    def __create__(columns=None, name=None, dtype=None):
        if columns is None:
            intrinsics.not_implemented()
        return MultiIndex._create_raw(columns, name, dtype)

    def __abstract__(self, context):
        return ValueHolder.AbstractViaPython()

    @staticmethod
    @hipy.raw
    def _create_raw(column, name, dtype, _context=None):
        return _context.wrap(MultiIndex(column, name, dtype))

    @hipy.compiled_function
    def __topython__(self):
        return pd.MultiIndex(self._cols, name=self.names, dtype=self.dtype)

    @hipy.compiled_function
    def __hipy__repr__(self):
        repr(intrinsics.to_python(self))

    @hipy.compiled_function
    def __str__(self):
        return str(intrinsics.to_python(self))

    @staticmethod
    def __hipy_create_type__() -> Type:
        return _NotRelevantType()

    def __hipy_get_type__(self) -> Type:
        return _NotRelevantType()

    @hipy.compiled_function
    def _update(self, table):
        return MultiIndex._create_raw([table.get_column(col) for col in self.names] , self.names, self.dtype)


    @hipy.compiled_function
    def _columns(self):
        r = {}
        for i in range(len(self._cols)):
            r[self.names[i]] = self._cols[i]
        return r



@hipy.classdef
class RangeIndex(Value):
    __HIPY_MUTABLE__ = False

    def __init__(self, start, stop, step, column):
        super().__init__(None)
        self.start = start
        self.stop = stop
        self.step = step
        self._column = column

    def __abstract__(self, context):
        return ValueHolder.AbstractViaPython()

    @staticmethod
    @hipy.raw
    def _create_raw(start, stop, step, column, _context=None):
        return _context.wrap(RangeIndex(start, stop, step, column))

    @hipy.compiled_function
    def __topython__(self):
        return pd.RangeIndex(self.start, self.stop, self.step)

    @hipy.compiled_function
    def __hipy__repr__(self):
        repr(intrinsics.to_python(self))

    @hipy.compiled_function
    def __str__(self):
        return str(intrinsics.to_python(self))

    @staticmethod
    def __hipy_create_type__() -> Type:
        return _NotRelevantType()

    def __hipy_get_type__(self) -> Type:
        return _NotRelevantType()

    @hipy.compiled_function
    def _update(self, table):
        return Index._create_raw(table.get_column("index"), None, "int64")

    @hipy.compiled_function
    def _columns(self):
        return {"index": self._column}

    @hipy.compiled_function
    def _lookup(self, key):
        return key - self.start


@hipy.classdef
class _iLocDFIndexer(Value):
    def __init__(self, df):
        super().__init__(None)
        self._df = df

    @hipy.compiled_function
    def __getitem__(self, item):
        if intrinsics.isa(item, slice):
            return self._df[item]
        elif intrinsics.isa(item, int):
            col_list = self._df._columns()
            val_list = [self._df[col].iloc[item] for col in col_list]
            return Series(val_list, Index(col_list), name=item)
        else:
            intrinsics.not_implemented()

    def __hipy_get_type__(self) -> Type:
        return _NotRelevantType()

    @staticmethod
    def __hipy_create_type__() -> Type:
        return _NotRelevantType()

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.to_python(self._df).iloc



@hipy.classdef
class DataFrameGroupBySeriesGroupBy(static_object["df","by","colname"]):
    def __init__(self, df, by,colname):
        super().__init__(lambda args: DataFrameGroupBySeriesGroupBy(*args), df, by,colname)


    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.to_python(self.df).groupby(self.by)[self.colname]

    @staticmethod
    @hipy.raw
    def __create__(df, by,colname,_context):
        return _context.wrap(DataFrameGroupBySeriesGroupBy(df, by,colname))

    @hipy.compiled_function
    def nunique(self):
        distinct_table= self.df._table.aggregate(self.by+[self.colname],[])
        raw_res=distinct_table.aggregate(self.by,[(self.colname, self.colname, 0, lambda x,y : x+1, lambda x:x)])
        index = MultiIndex([raw_res.get_column(k) for k in self.by], self.by)
        return Series._create_raw(raw_res.get_column(self.colname), index, name=self.colname)



@hipy.classdef
class DataFrameGroupBy(static_object["df","by"]):
    def __init__(self, df, by):
        super().__init__(lambda args: DataFrameGroupBy(*args), df, by)


    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.to_python(self.df).groupby(self.by)

    @staticmethod
    @hipy.raw
    def __create__(df, by,_context):
        return _context.wrap(DataFrameGroupBy(df, by))


    @hipy.compiled_function
    def agg(self,arg=None,**kwargs):
        def zero(col_type):
            if col_type==float:
                return 0.0
            elif col_type==int:
                return 0
            elif col_type == np.int64:
                return np.int64(0)
            elif col_type == np.float64:
                return np.float64(0.0)
            else:
                intrinsics.not_implemented()

        def min_val(col_type):
            if col_type==float:
                return 1.7976931348623157e308
            elif col_type==int:
                return 9223372036854775807
            elif col_type == np.int64:
                return np.int64(9223372036854775807)
            elif col_type == np.float64:
                return np.float64(1.7976931348623157e308)
            else:
                intrinsics.not_implemented()
        def handle_func(func, col_type):
            if func=="sum":
                return zero(col_type), lambda x,y: x+y, lambda x:x
            if func== "min":
                return min_val(col_type), lambda x,y: min(x,y), lambda x:x
            if func== "mean":
                return (zero(col_type),np.int64(0)), lambda x,y: (x[0]+y,x[1]+1), lambda x: x[0]/x[1]
            else:
                intrinsics.not_implemented()
        res = []
        if arg is None:
            pass
        elif intrinsics.isa(arg,_concrete_dict):
            for k in arg:
                input_col=k
                output_col=k
                init_val, agg_fn, finalize_fn = handle_func(arg[k],self.df._col_types[input_col])
                res.append((output_col,input_col,init_val,agg_fn,finalize_fn))
        else:
            intrinsics.not_implemented()
        for k in kwargs:
            output_col=k
            input_col=kwargs[k][0]
            init_val, agg_fn, finalize_fn = handle_func(kwargs[k][1],self.df._col_types[input_col])
            res.append((output_col,input_col,init_val,agg_fn,finalize_fn))
        raw_res = self.df._table.aggregate(self.by, res)
        raw_res = raw_res.sort(self.by,[True for _ in self.by])
        index = MultiIndex([raw_res.get_column(k) for k in self.by], self.by)
        return DataFrame._create_raw(raw_res, index)

    @hipy.compiled_function
    def __getitem__(self, item):
        if intrinsics.isa(item, str):
            return DataFrameGroupBySeriesGroupBy(self.df, self.by, item)
        else:
            intrinsics.not_implemented()


@hipy.classdef
class DataFrame(Value):
    def __init__(self, table, index, col_versions,col_types):
        super().__init__(None)
        self._table = table
        self.index = index
        self._col_versions = col_versions
        self._col_types=col_types

    def __abstract__(self, context):
        return ValueHolder.AbstractViaPython()

    @staticmethod
    @hipy.raw
    def _create_raw(table, index, _context):
        column_types=table.value._table_type.columns
        col_versions = _context.create_dict([_context.constant(c) for c,t in column_types],
                                            [_context.constant(0) for _ in column_types])
        col_types = _context.create_dict([_context.constant(c) for c, t in column_types],
                                         [_context.wrap(TypeValue(t)) for c, t in column_types])
        res = _context.wrap(DataFrame(table, index, col_versions, col_types))
        res.value.iloc = _context.wrap(_iLocDFIndexer(res))
        return res

    class DFType(Type):
        def __init__(self, index_columns: List[str], column_types: Dict[str, Type]):
            self.index_columns = index_columns
            self.column_types = column_types

        def ir_type(self):
            raise NotImplementedError()

        def construct(self, value, context):
            raise NotImplementedError()

        def __eq__(self, other):
            if not isinstance(other, DataFrame.DFType):
                return False
            return self.index_columns == other.index_columns and self.column_types == other.column_types

    @staticmethod
    def __hipy_create_type__(index_columns, column_types) -> Type:
        return _NotRelevantType()

    def __hipy_get_type__(self) -> Type:
        return _NotRelevantType()

    @staticmethod
    @hipy.compiled_function
    def __create__(data=None, index=None, columns=None, dtype=None, copy=None):
        intrinsics.only_implemented_if(data is not None, columns is None, dtype is None, copy is None)
        if intrinsics.isa(data, _concrete_dict):
            columns_dict = {k: column(data[k]) for k in data}
            if index is None:
                first_column = columns_dict[list(columns_dict)[0]]
                num_rows = len(first_column)
                index = RangeIndex._create_raw(0, num_rows, 1, column.sequential(num_rows))
            index_columns = index._columns()
            for k in index_columns:
                columns_dict[k] = index_columns[k]
            table_value = table(columns_dict)
            return DataFrame._create_raw(table_value, index)
        else:
            intrinsics.not_implemented()

    @staticmethod
    @hipy.compiled_function
    def from_dict(data):
        if intrinsics.isa(data, _concrete_dict):
            columns_dict = {k: column(data[k]) for k in data}
            first_column = columns_dict[list(columns_dict)[0]]
            num_rows = len(first_column)
            index = RangeIndex._create_raw(0, num_rows, 1, column.sequential(num_rows))
            index_columns = index._columns()
            index_column_names = list(index_columns)
            for k in index_columns:
                columns_dict[k] = index_columns[k]
            table_value = table(columns_dict)
            return DataFrame._create_raw(table_value, index)


    @hipy.compiled_function
    def _clone(self):
        return DataFrame._create_raw(self._table, self.index)

    @hipy.raw
    def _generate_pandas_metadata(self, _context):
        def get_pd_type(t):
            match t:
                case SimpleType(_ir_type=ir_type):
                    match ir_type:
                        case ir.IntType():
                            return "int64"
                        case ir.IntegerType(width=w):
                            return f"int{w}"
                        case ir.FloatType(width=w):
                            return f"float{w}"
                        case ir.StringType():
                            return "unicode"
                        case ir.BoolType():
                            return "bool"
                        case ir.PyObjType():
                            return "object"
                case hipy.lib.builtins.list.ListType(element_type=element_type):
                    return f"list[{get_pd_type(element_type)}]"
            raise NotImplementedError()

        def get_np_type(t):
            match t:
                case SimpleType(_ir_type=ir_type):
                    match ir_type:
                        case ir.IntType():
                            return "int64"
                        case ir.IntegerType(width=w):
                            return f"int{w}"
                        case ir.FloatType(width=w):
                            return f"float{w}"
                        case ir.StringType():
                            return "object"
                        case ir.BoolType():
                            return "bool"
                        case ir.PyObjType():
                            return "object"
                case hipy.lib.builtins.list.ListType(element_type=element_type):
                    return f"object"
            raise NotImplementedError()

        def get_type_metadata(t):
            match t:
                case SimpleType(_ir_type=ir_type):
                    match ir_type:
                        case ir.StringType():
                            return {"encoding": "UTF-8"}
                        case _:
                            return None
                case _:
                    return None

        res = {}
        res["columns"] = []
        for column_name, column_type in self.value._table.value._table_type.columns:
            pd_type = get_pd_type(column_type)
            np_type = get_np_type(column_type)
            metadata = get_type_metadata(column_type)
            res["columns"].append(
                {"name": None if column_name == "index" else column_name.replace("index", "",1),
                 "field_name": column_name,
                 "pandas_type": pd_type,
                 "numpy_type": np_type,
                 "metadata": metadata})
        virtual_index_columns=_context.perform_call(_context.get_attr(self.value.index, "_columns"),[])
        match virtual_index_columns:
            case ValueHolder(value=_concrete_dict(c_dict=c_dict)):
                res["index_columns"]=[]
                for k in c_dict.keys():
                    res["index_columns"].append(k)
            case _:
                raise RuntimeError("Expected a list")
        return _context.constant(json.dumps(res))

    @hipy.compiled_function
    def __topython__(self):
        pyarrow_table = intrinsics.to_python(self._table)
        pyarrow_table = pyarrow_table.replace_schema_metadata(
            {"pandas": self._generate_pandas_metadata()})
        return pyarrow_table.to_pandas()

    @hipy.compiled_function
    def __hipy__repr__(self):
        return repr(self.__topython__())

    @hipy.compiled_function
    def __str__(self):
        return str(self.__topython__())

    @hipy.compiled_function
    def _columns(self):
        index_cols=self.index._columns()
        table_cols=self._table._column_types
        return [col for col in table_cols if col not in index_cols]

    @hipy.compiled_function
    def sort_values(self, by, ascending=True, axis=0):
        if axis != 0:
            intrinsics.not_implemented()
        if intrinsics.isa(by, str):
            by = [by]
        if intrinsics.isa(ascending, bool):
            ascending = [ascending for i in range(len(by))]
        new_table=self._table.sort(by, ascending)
        return DataFrame._create_raw(new_table, self.index._update(new_table))

    @hipy.compiled_function
    def __getitem__(self, key):
        if intrinsics.isa(key, _const_str):
            col = self._table.get_column(key)
            return Series._create_raw_df(col, self.index, self, key, self._col_versions[key])
        elif intrinsics.isa(key, Series):
            if key._element_type== bool:
                new_table = self._table.filter_by_column(key._data)
                return DataFrame._create_raw(new_table, self.index._update(new_table))
            else:
                intrinsics.not_implemented()
        elif intrinsics.isa(key, list):
            if intrinsics.isa(key,_concrete_list) and key._element_type==str:
                new_table = self._table.select_columns(key+list(self.index._columns()))
                return DataFrame._create_raw(new_table, self.index._update(new_table))
        elif intrinsics.isa(key, slice):
            new_table=self._table.get_slice(key)
            return DataFrame._create_raw(new_table, self.index._update(new_table))
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __setitem__(self, key, value):
        if intrinsics.isa(key, _const_str):
            if intrinsics.isa(value, Series):
                if key in self._col_versions:
                    # case 1: column already exists -> update version and set new column
                    version = self._col_versions[key]
                    self._col_versions[key] = version + 1
                    new_table = self._table.set_column(key, value._data)
                    self._table = new_table
                else:
                    # case 2: column does not exist -> add new column
                    self._col_versions[key] = 0
                    self._col_types[key] = value._element_type
                    new_table = self._table.set_column(key, value._data)
                    self._table = new_table
            else:
                intrinsics.not_implemented()
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __hipy_getattr__(self, key):
        if key == "columns":
            return self._columns()
        else:

            # todo: check if column exists
            return self[key]

    @hipy.raw
    def __hipy_setattr__(self, key, value, _context):
        match key:
            case ValueHolder(value=_const_str(cval=attr)):
                match attr:
                    case "_table":
                        self.value._table = value
                    case _:
                        raise NotImplementedError("Attribute not found")
            case _:
                raise NotImplementedError("Attribute not found")
    @staticmethod
    @hipy.compiled_function
    def _row_to_series(row,col_list,index_col_list):
        col_values = [_to_python_type(row[col]) for col in col_list]
        name=None
        if len(index_col_list)>1:
            intrinsics.not_implemented()
        else:
            name=row[index_col_list[0]]
        return Series(col_values, Index(col_list), name=name)
    @hipy.compiled_function
    def apply(self, func, axis=0):
        if axis == 0:
            res = {}
            for col in self._col_types:
                curr_series = self[col]
                res[col] = func(curr_series)._data
            index_cols = self.index._columns()
            for col in index_cols:
                res[col] = index_cols[col]
            return DataFrame._create_raw(table(res), self.index)
        elif axis == 1:
            col_list = intrinsics.as_constant(self._columns())
            index_col_list = intrinsics.as_constant(list(self.index._columns()))
            res_col= self._table.apply_row_wise(lambda row: func(DataFrame._row_to_series(row,col_list,index_col_list)))
            return Series._create_raw(res_col, self.index)
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def merge(self,right, how='inner', on=None, left_on=None, right_on=None):
        return hipy.lib.pandas.merge(self, right, how=how, on=on, left_on=left_on, right_on=right_on)

    @hipy.compiled_function
    def fillna(self, value):
        res = self._clone()
        for col in self._col_types:
            col_type=self._col_types[col]
            if col_type==float:
                res[col]=res[col].fillna(value)
            elif col_type==np.float64:
                res[col]=res[col].fillna(np.float64(value))

        return res

    @hipy.compiled_function
    def groupby(self, by):
        return DataFrameGroupBy(self, by)

    @hipy.compiled_function
    def reset_index(self):
        all_cols=list(self._table._column_types)
        index_col_name="index"
        if index_col_name in all_cols:
            index_col_name="level_0"
        num_rows = len(self._table)
        index = RangeIndex._create_raw(0, num_rows, 1, column.sequential(num_rows))
        res_table = self._table.set_column(index_col_name, index._column)
        return DataFrame._create_raw(res_table, index)

@hipy.compiled_function
def merge(left,right, how='inner', on=None, left_on=None, right_on=None,left_index=False, right_index=False):
    if intrinsics.isa(left, DataFrame) and intrinsics.isa(right, DataFrame):
        left_cols = left._columns()
        right_cols = right._columns()
        left_index_cols = left.index._columns()
        right_index_cols = right.index._columns()
        left_res_cols = []
        right_res_cols = []
        res_index_cols = []
        if on is not None:
            if intrinsics.isa(on, str):
                on = [on]
            left_on = on
            right_on = on
            left_res_cols=[ c for c in left_cols if c not in left_index_cols]
            right_res_cols= [col for col in right_cols if col not in on and col not in right_index_cols]
        elif left_on is not None and right_on is not None:
            if intrinsics.isa(left_on, str):
                left_on = [left_on]
            if intrinsics.isa(right_on, str):
                right_on = [right_on]
            left_res_cols = [col for col in left_cols if col not in left_index_cols]
            right_res_cols = [col for col in right_cols if col not in right_index_cols]
        elif left_index and right_index:
            left_on = list(left_index_cols)
            right_on = list(right_index_cols)
            left_res_cols = list(left_cols)+list(left_index_cols)
            right_res_cols = [col for col in right_cols+list(right_index_cols) if col not in left_res_cols]
            res_index_cols= [col for col in left_index_cols]+[col for col in right_index_cols if col not in left_index_cols]

        else:
            intrinsics.not_implemented()

        if how == "inner":
            res_table = left._table.join_inner(right._table, left_on, right_on, left_res_cols, right_res_cols)
            if len(res_index_cols)>0:
                index=MultiIndex([res_table.get_column(col) for col in res_index_cols],res_index_cols)
                return DataFrame._create_raw(res_table, index)
            else:
                num_rows = len(res_table)
                index = RangeIndex._create_raw(0, num_rows, 1, column.sequential(num_rows))
                res_table = res_table.set_column("index", index._column)
                return DataFrame._create_raw(res_table, index)
        elif how == "left":
            for col in right_res_cols:
                col_type= right._col_types[col]
                if col_type == int or col_type == np.int64:
                    right[col] = right[col].apply(lambda x: np.float64(x))
                elif col_type != float or col_type != np.float64:
                    intrinsics.not_implemented()
            res_table = left._table.join_left(right._table, left_on, right_on, left_res_cols, right_res_cols)
            num_rows = len(res_table)
            index = RangeIndex._create_raw(0, num_rows, 1, column.sequential(num_rows))
            res_table = res_table.set_column("index", index._column)
            return DataFrame._create_raw(res_table, index)
        else:
            intrinsics.not_implemented()

    else:
        intrinsics.not_implemented()




@hipy.classdef
class _StringMethods(Value):
    def __init__(self, series):
        super().__init__(None)
        self._series = series

    @hipy.compiled_function
    def len(self):
        return self._series.apply(lambda x: x.__len__())

    @hipy.compiled_function
    def lower(self):
        return self._series.apply(lambda x: x.lower())

    @hipy.compiled_function
    def upper(self):
        return self._series.apply(lambda x: x.upper())

    @hipy.compiled_function
    def contains(self, pat):
        return self._series.apply(lambda x: pat in x)

    @hipy.compiled_function
    def slice(self, start=None, stop=None, step=None):
        return self._series.apply(lambda x: x[start:stop:step])

    def __hipy_get_type__(self) -> Type:
        return _NotRelevantType()

    @staticmethod
    def __hipy_create_type__() -> Type:
        return _NotRelevantType()

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.to_python(self._series).str


@hipy.classdef
class _DateMethods(Value):
    def __init__(self, series):
        super().__init__(None)
        self._series = series

    @hipy.compiled_function
    def __hipy_getattr__(self, item):
        if item=="year":
            return self._series.apply(lambda x: x.year)
        else:
            intrinsics.not_implemented()



    def __hipy_get_type__(self) -> Type:
        return _NotRelevantType()

    @staticmethod
    def __hipy_create_type__() -> Type:
        return _NotRelevantType()

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.to_python(self._series).dt


@hipy.classdef
class _iLocSeriesIndexer(Value):
    def __init__(self, series):
        super().__init__(None)
        self._series = series

    @hipy.compiled_function
    def __getitem__(self, item):
        if intrinsics.isa(item, slice):
            return self._series[item]
        elif intrinsics.isa(item, int):
            return self._series._data.get_by_index(item)
        else:
            intrinsics.not_implemented()

    def __hipy_get_type__(self) -> Type:
        return _NotRelevantType()

    @staticmethod
    def __hipy_create_type__() -> Type:
        return _NotRelevantType()

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.to_python(self._series).iloc


@hipy.classdef
class Series(Value):
    def __init__(self, index, data_column=None, df=None, name=None, version=None, concrete_values=None):
        super().__init__(None)
        self._data_ = data_column
        self.index = index
        self._df = df
        self.name = name
        self._version = version
        assert concrete_values is not None
        self._concrete_values = concrete_values

    @hipy.compiled_function
    def __hipy_getattr__(self, item):
        if item == "_element_type":
            return self._data._element_type
        elif item == "_data":
            if self._df is None:
                if self._concrete_values is not None:
                    self._data_ = column(self._concrete_values)
                return self._data_
            else:
                df_col_versions = self._df._col_versions
                if self.name in df_col_versions:
                    version = df_col_versions[self.name]
                    if version == self._version:
                        return self._df._table.get_column(self.name)
                    else:
                        return self._data_
                else:
                    return self._data_
        else:
            intrinsics.not_implemented()
    @hipy.raw
    def __hipy_setattr__(self, key, value,_context):
        match key:
            case ValueHolder(value=_const_str(cval=attr)):
                match attr:
                    case "_data_":
                        self.value._data_ = value
                    case _:
                        raise NotImplementedError("Attribute not found")
            case _:
                raise NotImplementedError("Attribute not found")

    def __abstract__(self, context):
        return ValueHolder.AbstractViaPython()

    @staticmethod
    @hipy.compiled_function
    def __create__(data=None, index=None,name=None):
        if data is None:
            intrinsics.not_implemented()
        if data._element_type == int or data._element_type == float:
            data= [_to_native_type(val) for val in data]
        concrete_values = None
        if intrinsics.isa(data, _concrete_list):
            concrete_values = [val for val in data]  # copy necessary
        data_column = None
        data_length=len(data)
        if concrete_values is None:
            data_column = column(data)
        if index is None:
            index = RangeIndex._create_raw(0, data_length, 1, column.sequential(data_length))
        return Series._create_raw(data_column, index, concrete_values=concrete_values,name=name)

    @staticmethod
    @hipy.raw
    def _create_raw(data_column, index, concrete_values=None, _context=None,name=None):
        if concrete_values is None:
            concrete_values= _context.constant(None)
        if name is None:
            name= _context.constant(None)
        res = _context.wrap(Series(index, data_column, name=name, df=_context.constant(None),
                                   concrete_values=concrete_values))
        res.value.str = _context.wrap(_StringMethods(res))
        res.value.dt = _context.wrap(_DateMethods(res))
        res.value.iloc = _context.wrap(_iLocSeriesIndexer(res))
        return res

    @staticmethod
    @hipy.raw
    def _create_raw_df(data_column, index, df, name, version, _context):
        res = _context.wrap(Series(index, data_column=data_column, df=df, name=name, version=version,
                                   concrete_values=_context.constant(None)))
        res.value.str = _context.wrap(_StringMethods(res))
        res.value.dt = _context.wrap(_DateMethods(res))
        res.value.iloc = _context.wrap(_iLocSeriesIndexer(res))
        return res


    @staticmethod
    def __hipy_create_type__(element_type) -> Type:
        return _NotRelevantType()

    def __hipy_get_type__(self) -> Type:
        return _NotRelevantType()

    @hipy.compiled_function
    def __topython__(self):
        index = intrinsics.to_python(self.index)
        if self._concrete_values is not None:
            return pd.Series(self._concrete_values,index,name=self.name)
        else:
            pyarrow_data = intrinsics.to_python(self._data)
            res = pyarrow_data.to_pandas()
            res.index = index
            res.name=self.name
            return res

    @hipy.compiled_function
    def _element_wise(self, other, fn):
        if intrinsics.isa(other,object):
            intrinsics.not_implemented()
        elif intrinsics.isa(other, Series):
            res_col = self._data.element_wise(other._data, lambda a,b: _to_native_type(fn(a,b)))
            return Series._create_raw(res_col, self.index)
        else:
            res_col = self._data.apply(lambda x: _to_native_type(fn(x, other)))
            return Series._create_raw(res_col, self.index)

    @hipy.compiled_function
    def __add__(self, other):
        return self._element_wise(other, lambda a, b: a + b)
    @hipy.compiled_function
    def __sub__(self, other):
        return self._element_wise(other, lambda a, b: a - b)
    @hipy.compiled_function
    def __mul__(self, other):
        return self._element_wise(other, lambda a, b: a * b)
    @hipy.compiled_function
    def __truediv__(self, other):
        return self._element_wise(other, lambda a, b: a / b)
    @hipy.compiled_function
    def add(self,other):
        return self+other
    @hipy.compiled_function
    def sub(self,other):
        return self-other
    @hipy.compiled_function
    def mul(self,other):
        return self*other
    @hipy.compiled_function
    def div(self,other):
        return self/other

    @hipy.compiled_function
    def __eq__(self, other):
        return self._element_wise(other, lambda a, b: a == b)

    @hipy.compiled_function
    def __ne__(self, other):
        return self._element_wise(other, lambda a, b: a != b)

    @hipy.compiled_function
    def __lt__(self, other):
        return self._element_wise(other, lambda a, b: a < b)

    @hipy.compiled_function
    def __le__(self, other):
        return self._element_wise(other, lambda a, b: a <= b)

    @hipy.compiled_function
    def __gt__(self, other):
        return self._element_wise(other, lambda a, b: a > b)

    @hipy.compiled_function
    def __ge__(self, other):
        return self._element_wise(other, lambda a, b: a >= b)

    @hipy.compiled_function
    def __and__(self, other):
        return self._element_wise(other, lambda a, b: a & b)

    @hipy.compiled_function
    def __or__(self, other):
        return self._element_wise(other, lambda a, b: a | b)

    @hipy.compiled_function
    def __invert__(self):
        return self.apply(lambda x: ~x)

    @hipy.compiled_function
    def apply(self, fn):
        res_col = self._data.apply(lambda x: _to_native_type(fn(_to_python_type(x))))
        return Series._create_raw(res_col, self.index)

    @hipy.compiled_function
    def __hipy__repr__(self):
        return repr(self.__topython__())

    @hipy.compiled_function
    def __str__(self):
        return str(self.__topython__())

    @hipy.raw
    def _const_position(self,pos,_context):
        if self.value._concrete_values is not None and not isinstance(self.value._concrete_values.value,VoidValue):
            match pos:
                case ValueHolder(value=CValue(cval=cval)):
                    try:
                        return self.value._concrete_values.value.items[cval]
                    except ValueError as e:
                        pass
        return _context.constant(None)
    @hipy.compiled_function
    def _is_native(self):
        return self._element_type == np.int64 or self._element_type == np.float64

    @hipy.compiled_function
    def mask(self,cond, other=None):
        if other is None:
            intrinsics.not_implemented()
        if self._is_native():
            other=_to_native_type(other)
        return self._element_wise(cond, lambda curr, cond: other if cond else curr)

    @hipy.compiled_function
    def __getitem__(self, item):
        if intrinsics.isa(item, slice):
            cols = {'__col__': self._data}
            index_cols = self.index._columns()
            for col in index_cols:
                cols[col] = index_cols[col]
            tmp_table = table(cols).get_slice(item)
            return Series._create_raw(tmp_table.get_column("__col__"),
                                      self.index._update(tmp_table))
        else:
            position = self.index._lookup(item)
            if intrinsics.isa(position, int):
                const_element = self._const_position(position)
                if const_element is not None:
                    return const_element
                else:
                    return self._data.get_by_index(position)
            elif intrinsics.isa(position, column):
                cols = {'__col__': self._data}
                index_cols = self.index._columns()
                for col in index_cols:
                    cols[col] = index_cols[col]
                tmp_table = table(cols).filter_by_column(position)
                if (len(tmp_table) == 1):
                    return tmp_table.get_column("__col__").get_by_index(0)
                else:
                    return Series._create_raw(tmp_table.get_column("__col__"), self.index._update(tmp_table))
            else:
                intrinsics.not_implemented()
    @staticmethod
    @hipy.compiled_function
    def _sum_combine(left, right):
        if left[0] and right[0]:
            return (True, left[1] + right[1])
        elif left[0]:
            return left
        else:
            return right
    @hipy.compiled_function
    def sum(self):
        undef = intrinsics.undef(self._element_type)
        initial_value = (False, undef)
        res=self._data.aggregate(initial_value,lambda a, b: (True, a[1] + b) if a[0] else (True, b), Series._sum_combine)
        if res[0]:
            return res[1]
        else:
            if self._element_type==np.float64:
                return np.float64(0.0)
            elif self._element_type==np.int64:
                return np.int64(0)
            else:
                return 0
    @hipy.compiled_function
    def fillna(self, value):
        if intrinsics.isa(value, self._element_type):
            return self.apply(lambda x: value if np.isnan(np.float64(x)) else x)
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def reset_index(self):
        cols = {self.name: self._data}
        index_cols = self.index._columns()
        for col in index_cols:
            cols[col] = index_cols[col]
        new_index_name="index"
        if new_index_name in cols:
            new_index_name="level_0"
        num_rows = len(self._data)
        new_index = RangeIndex._create_raw(0, num_rows, 1, column.sequential(num_rows))
        cols[new_index_name] = new_index._column
        new_table = table(cols)

        return DataFrame._create_raw(new_table, new_index)

@hipy.classdef
class Timestamp(Value):
    def __init__(self, value):
        super().__init__(value)
    @staticmethod
    @hipy.compiled_function
    def __create__(ts_input,unit='ns'):
        if unit != 'ns':
            intrinsics.not_implemented()
        if intrinsics.isa(ts_input, int):
            return intrinsics.reinterpret(ts_input, Timestamp)
        elif intrinsics.isa(ts_input, str):
            return intrinsics.call_builtin("date.parse", Timestamp,[ts_input])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __topython__(self):
        return pd.Timestamp(intrinsics.reinterpret(self, int), unit='ns')


    def __hipy_get_type__(self) -> Type:
        return SimpleType(Timestamp,ir.i64)

    @staticmethod
    def __hipy_create_type__() -> Type:
        return SimpleType(Timestamp,ir.i64)

    @hipy.compiled_function
    def __hipy_getattr__(self, key):
        if key=="hour":
            return intrinsics.call_builtin("date.get_hour",int,[self])
        elif key=="year":
            return intrinsics.call_builtin("date.get_year",int,[self])
        else:
            intrinsics.not_implemented()


@hipy.compiled_function
def to_datetime(s):
    if intrinsics.isa(s, Series):
        return s.apply(lambda x: Timestamp(x))
    else:
        intrinsics.not_implemented()


