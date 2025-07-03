__HIPY_MODULE__ = "numpy"

import numpy as _np
import sys

import hipy
from hipy import intrinsics, ir
from hipy.lib.builtins import _const_int, _const_float
from hipy.value import SimpleType, Value, Type, raw_module, ValueHolder, TypeValue, HLCFunctionValue

hipy.register(sys.modules[__name__])

np = raw_module(_np)


@hipy.classdef
class int64(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)

    def __hipy_get_type__(self):
        return SimpleType(int64, ir.i64)

    def __hipy_create_type__(*args) -> Type:
        return SimpleType(int64, ir.i64)

    @hipy.compiled_function
    def _int_op(self, op, other, reverse=False):
        other = _to_numpy(other)
        if intrinsics.isa(other, int64):  # todo: handle int32,...
            left = other if reverse else self
            right = self if reverse else other
            return intrinsics.call_builtin("scalar.int."+op, int64, [left, right])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def _cmp_op(self, op, other):
        other = _to_numpy(other)
        if intrinsics.isa(other, int64):
            return intrinsics.call_builtin("scalar.int.compare."+op, bool, [self, other])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __add__(self, other):
        return self._int_op("add", other)

    @hipy.compiled_function
    def __radd__(self, other):
        return self._int_op("add", other, reverse=True)

    @hipy.compiled_function
    def __sub__(self, other):
        return self._int_op("sub", other)

    @hipy.compiled_function
    def __rsub__(self, other):
        return self._int_op("sub", other, reverse=True)

    @hipy.compiled_function
    def __mul__(self, other):
        return self._int_op("mul", other)

    @hipy.compiled_function
    def __rmul__(self, other):
        return self._int_op("mul", other, reverse=True)

    @hipy.compiled_function
    def __eq__(self, other):
        return self._cmp_op("eq", other)

    @hipy.compiled_function
    def __ne__(self, other):
        return self._cmp_op("neq", other)

    @hipy.compiled_function
    def __lt__(self, other):
        return self._cmp_op("lt", other)

    @hipy.compiled_function
    def __le__(self, other):
        return self._cmp_op("lte", other)

    @hipy.compiled_function
    def __gt__(self, other):
        return self._cmp_op("gt", other)

    @hipy.compiled_function
    def __ge__(self, other):
        return self._cmp_op("gte", other)

    @hipy.compiled_function
    def __truediv__(self, other):
        if intrinsics.isa(other, int):
            return float(self) / float(other)
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __rtruediv__(self, other):
        if intrinsics.isa(other, int):
            return float(other) / float(self)
        elif intrinsics.isa(other, float):
            return float64(other) / self
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __mod__(self, other):
        return self._int_op("mod", other)

    @hipy.compiled_function
    def __rmod__(self, other):
        return self._int_op("mod", other, reverse=True)

    @staticmethod
    @hipy.compiled_function
    def __create__(value):
        if intrinsics.isa(value, _const_int):
            return _const_int64(value)
        elif intrinsics.isa(value, int):
            return intrinsics.call_builtin("scalar.int.pyint_to_int64", int64, [value])
        elif intrinsics.isa(value, float):
            return int64(int(value))
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __topython__(self):
        return np.int64(intrinsics.call_builtin("scalar.int.to_python", object, [self]))

    @hipy.compiled_function
    def __hipy__repr__(self):
        return intrinsics.call_builtin("scalar.int.to_string", str, [self])

    @hipy.compiled_function
    def __float__(self):
        return intrinsics.call_builtin("scalar.float.from_int", float, [self])
    @hipy.compiled_function
    def __int__(self):
        return intrinsics.call_builtin("scalar.int.int64_to_pyint", int, [self])

    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, int64):
            return self, other, lambda val: int64(val)
        else:
            raise NotImplementedError()

@hipy.classdef
class _const_int64(int64):
    __HIPY_MATERIALIZED__ = False
    def __init__(self, cval):
        int64.__init__(self, None)
        self._cval=cval

    @staticmethod
    @hipy.raw
    def __create__(value,_context):
        match value:
            case ValueHolder(value=_const_int(cval=cval)):
                return _context.wrap(_const_int64(cval))
            case _:
                raise NotImplementedError()
    def __abstract__(self, _context):
        return int64(ir.Constant(_context.block, self._cval, ir.i64).result)
    @hipy.raw
    def __int__(self,_context):
        return _context.constant(self.value._cval)


@hipy.classdef
class float64(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False
    def __init__(self, value):
        super().__init__(value)

    def __hipy_get_type__(self):
        return SimpleType(float64, ir.f64)

    def __hipy_create_type__(*args) -> Type:
        return SimpleType(float64, ir.f64)

    @hipy.compiled_function
    def _float_op(self, op, other, reverse=False):
        other = _to_numpy(other)
        if intrinsics.isa(other, float64):  # todo: handle int32,...
            left = other if reverse else self
            right = self if reverse else other
            return intrinsics.call_builtin("scalar.float."+op, float64, [left, right])
        elif intrinsics.isa(other, int64):
            return self._float_op(op, float64(other), reverse=reverse)
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def _cmp_op(self, op, other):
        other = _to_numpy(other)
        if intrinsics.isa(other, float64):
            return intrinsics.call_builtin("scalar.float.compare."+op, bool, [self, other])
        elif intrinsics.isa(other, int64):
            return self._cmp_op(op, float64(other))
        else:
            intrinsics.not_implemented()

    @staticmethod
    @hipy.compiled_function
    def __create__(value):
        if intrinsics.isa(value, float64):
            return value
        elif intrinsics.isa(value, _const_float) or intrinsics.isa(value, _const_int):
            return _const_float64(value)
        elif intrinsics.isa(value, float):
            return intrinsics.reinterpret(value, float64)
        elif intrinsics.isa(value, int):
            return float64(float(value))
        elif intrinsics.isa(value, int64):
            return intrinsics.call_builtin("scalar.float.from_int", float64, [value])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __float__(self):
        return intrinsics.reinterpret(self, float)

    @hipy.compiled_function
    def __int__(self):
        return intrinsics.call_builtin("scalar.float.to_int", int, [self])

    @hipy.compiled_function
    def __add__(self, other):
        return self._float_op("add", other)

    @hipy.compiled_function
    def __radd__(self, other):
        return self._float_op("add", other, reverse=True)

    @hipy.compiled_function
    def __sub__(self, other):
        return self._float_op("sub", other)

    @hipy.compiled_function
    def __rsub__(self, other):
        return self._float_op("sub", other, reverse=True)

    @hipy.compiled_function
    def __mul__(self, other):
        return self._float_op("mul", other)

    @hipy.compiled_function
    def __rmul__(self, other):
        return self._float_op("mul", other, reverse=True)

    @hipy.compiled_function
    def __truediv__(self, other):
        return self._float_op("div", other)

    @hipy.compiled_function
    def __rtruediv__(self, other):
        return self._float_op("div", other, reverse=True)

    @hipy.compiled_function
    def __mod__(self, other):
        return self._float_op("mod", other)

    @hipy.compiled_function
    def __rmod__(self, other):
        return self._float_op("mod", other, reverse=True)

    @hipy.compiled_function
    def __pow__(self, other):
        if intrinsics.isa(other, _const_int) or intrinsics.isa(other, _const_int64):
            if int(other) < 5:
                res = self
                for i in range(1, int(other)):
                    res = res * self
                return res
        return self._float_op("pow", other)

    @hipy.compiled_function
    def __eq__(self, other):
        return self._cmp_op("eq", other)

    @hipy.compiled_function
    def __ne__(self, other):
        return self._cmp_op("neq", other)

    @hipy.compiled_function
    def __lt__(self, other):
        return self._cmp_op("lt", other)

    @hipy.compiled_function
    def __le__(self, other):
        return self._cmp_op("lte", other)

    @hipy.compiled_function
    def __gt__(self, other):
        return self._cmp_op("gt", other)

    @hipy.compiled_function
    def __ge__(self, other):
        return self._cmp_op("gte", other)

    @hipy.compiled_function
    def __topython__(self):
        return np.float64(intrinsics.call_builtin("scalar.float.to_python", object, [self]))

    @hipy.compiled_function
    def __hipy__repr__(self):
        return intrinsics.call_builtin("scalar.float.to_string", str, [self])

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, float64):
            return self, other, lambda val: float64(val)
        else:
            raise NotImplementedError()

@hipy.classdef
class _const_float64(float64):
    __HIPY_MATERIALIZED__ = False
    def __init__(self, cval):
        super().__init__(None)
        self._cval=cval

    @staticmethod
    @hipy.raw
    def __create__(value,_context):
        match value:
            case ValueHolder(value=_const_float(cval=cval)):
                return _context.wrap(_const_float64(cval))
            case ValueHolder(value=_const_int(cval=cval)):
                return _context.wrap(_const_float64(float(cval)))
            case _:
                raise NotImplementedError()
    def __abstract__(self, _context):
        return float64(ir.Constant(_context.block, self._cval, ir.f64).result)

@hipy.compiled_function
def _to_numpy(x):
    if intrinsics.isa(x, float64):  # or intrinsics.isa(x, float32):
        return x
    elif intrinsics.isa(x, int64):
        return x
    elif intrinsics.isa(x, int):
        return int64(x)
    elif intrinsics.isa(x, float):
        return float64(x)
    elif intrinsics.isa(x, ndarray):
        return x
    else:
        intrinsics.not_implemented()


@hipy.compiled_function
def _array_binary_element_wise(array1, array2, fn):
    bound_fn = intrinsics.bind(fn, [array1._dtype, array2._dtype])
    #todo: broadcast
    res = intrinsics.call_builtin("array.binary_op", intrinsics.create_type(ndarray, bound_fn.res_type, array1.shape),
                                   [array1, array2, bound_fn], side_effects=False)
    res.shape=array1.shape
    return res


@hipy.compiled_function
def _array_apply_scalar(array, fn):
    bound_fn = intrinsics.bind(fn, [array._dtype])
    res = intrinsics.call_builtin("array.apply_scalar", intrinsics.create_type(ndarray, bound_fn.res_type, array.shape),
                                   [array, bound_fn],side_effects=False)
    res.shape=array.shape
    return res

@hipy.compiled_function
def _float_function(op, x):
    x = _to_numpy(x)
    if intrinsics.isa(x, float64):
        return intrinsics.call_builtin("scalar.float."+op, float64, [x])
    elif intrinsics.isa(x, int64):
        z = float64(x)
        return _float_function(op, z)
    elif intrinsics.isa(x, ndarray):
        return _array_apply_scalar(x, lambda x: _float_function(op, x))
    else:
        intrinsics.not_implemented()
@hipy.compiled_function
def sin(x):
    return _float_function("sin", x)

@hipy.compiled_function
def cos(x):
    return _float_function("cos", x)

@hipy.compiled_function
def arcsin(x):
    return _float_function("arcsin", x)

@hipy.compiled_function
def sqrt(x):
    return _float_function("sqrt", x)

@hipy.compiled_function
def log(x):
    return _float_function("log", x)

@hipy.compiled_function
def exp(x):
    return _float_function("exp", x)

@hipy.compiled_function
def _convert_to_dtype(value, expected_dtype):
    if intrinsics.typeof(value)==expected_dtype:
        return value
    else:
        return expected_dtype(value)

@hipy.compiled_function
def _convert_to_np_type(value):
    if intrinsics.isa(value, float):
        return float64(value)
    elif intrinsics.isa(value, int):
        return int64(value)
    else:
        return value


@hipy.classdef
class ndarray(Value):

    def __init__(self, value, dtype, shape, shape_value=None):
        super().__init__(value)
        self._dtype = dtype
        self._shape = shape
        self.shape = shape_value
    class ArrayType(Type):
        def __init__(self, dtype, shape):
            self._dtype = dtype
            self._shape = shape

        def ir_type(self):
            if isinstance(self._dtype, ValueHolder):
                self._dtype = self._dtype.value.type
            return ir.ArrayType(self._dtype.ir_type(), self._shape)

        def construct(self, value, context):
            shape_vals=[]
            for i,s in enumerate(self._shape):
                if s is None:
                    shape_vals.append(context.wrap(hipy.lib.builtins.int(ir.CallBuiltin(context.block, "array.dim", [value, context.constant(i).get_ir_value()], ir.int, side_effects=False).result)))
                else:
                    shape_vals.append(context.constant(s))
            shape_value=context.wrap(hipy.lib.builtins.tuple(shape_vals))
            return ndarray(value, self._dtype, self._shape,shape_value)

        def __eq__(self, other):
            if isinstance(other, ndarray.ArrayType):
                return self._dtype == other._dtype and self._shape == other._shape
            else:
                return False

        def __repr__(self):
            return f"array.ArrayType({self._dtype}, {self._shape})"

    def __hipy_get_type__(self):
        return ndarray.ArrayType(self._dtype, self._shape)

    @staticmethod
    def __hipy_create_type__(dtype, shape) -> Type:
        assert dtype is not None
        assert shape is not None
        return ndarray.ArrayType(dtype, shape)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.call_builtin("array.to_python", object, [self])

    @hipy.compiled_function
    def __hipy__repr__(self):
        # this avoids going through the context -> no sideeffects
        return repr(intrinsics.call_builtin("array.to_python", object, [self]))

    @hipy.compiled_function
    def __str__(self):
        return str(intrinsics.call_builtin("array.to_python", object, [self]))

    @hipy.compiled_function
    def _element_wise(self, other, fn):
        other = _to_numpy(other)
        if intrinsics.isa(other, ndarray):
            return _array_binary_element_wise(self, other, fn)
        else:
            return _array_apply_scalar(self, lambda x: fn(x, other))

    @hipy.compiled_function
    def __add__(self, other):
        return self._element_wise(other, lambda x, y: x + y)

    @hipy.compiled_function
    def __radd__(self, other):
        return self._element_wise(other, lambda x, y: y + x)

    @hipy.compiled_function
    def __sub__(self, other):
        return self._element_wise(other, lambda x, y: x - y)

    @hipy.compiled_function
    def __rsub__(self, other):
        return self._element_wise(other, lambda x, y: y - x)

    @hipy.compiled_function
    def __mul__(self, other):
        return self._element_wise(other, lambda x, y: x * y)

    @hipy.compiled_function
    def __rmul__(self, other):
        return self._element_wise(other, lambda x, y: y * x)

    @hipy.compiled_function
    def __truediv__(self, other):
        return self._element_wise(other, lambda x, y: x / y)

    @hipy.compiled_function
    def __rtruediv__(self, other):
        return self._element_wise(other, lambda x, y: y / x)

    @hipy.compiled_function
    def __pow__(self, other):
        return self._element_wise(other, lambda x, y: x ** y)

    @hipy.compiled_function
    def __eq__(self, other):
        return self._element_wise(other, lambda x, y: x == y)
    @hipy.compiled_function
    def __ne__(self, other):
        return self._element_wise(other, lambda x, y: x != y)
    @hipy.compiled_function
    def __lt__(self, other):
        return self._element_wise(other, lambda x, y: x < y)
    @hipy.compiled_function
    def __le__(self, other):
        return self._element_wise(other, lambda x, y: x <= y)
    @hipy.compiled_function
    def __gt__(self, other):
        return self._element_wise(other, lambda x, y: x > y)
    @hipy.compiled_function
    def __ge__(self, other):
        return self._element_wise(other, lambda x, y: x >= y)

    @hipy.compiled_function
    def __getitem__(self, item):
        if intrinsics.isa(item, int):
            item= (item,)
        if intrinsics.isa(item, slice):
            item= (item,)
        if intrinsics.isa(item,tuple):
            only_ints= True
            for i in item:
                if intrinsics.isa(i, int):
                    pass
                elif intrinsics.isa(i, slice):
                    only_ints=False
            idx_list=[]
            exactly_shape=len(item)==len(self.shape)

            for i in item:
                if intrinsics.isa(i,slice):
                    stop = i.stop if i.stop is not None else self.shape[len(idx_list)]
                    stop = stop if stop>=0 else self.shape[len(idx_list)]+stop
                    start = i.start if i.start is not None else 0
                    start = start if start>=0 else self.shape[len(idx_list)]+start
                    step = i.step if i.step is not None else 1
                    i = (start, stop, step)
                idx_list.append(i)
            if only_ints and exactly_shape:
                return intrinsics.call_builtin("array.get", self._dtype, [self]+idx_list)
            else:
                shape_list=[]
                for i in range(len(self.shape)):
                    if i<len(item):
                        if intrinsics.isa(item[i], int):
                            pass
                        else:
                            shape_list.append(None)
                    else:
                        shape_list.append(None)


                if not exactly_shape:
                    for i in range(len(item),len(self.shape)):
                        idx_list.append((0, self.shape[i], 1))
                return intrinsics.call_builtin("array.create_view", intrinsics.create_type(ndarray, self._dtype, shape_list), [self]+idx_list)
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __setitem__(self, key, value):
        if intrinsics.isa(key, int):
            key= (key,)
        if intrinsics.isa(key, slice):
            key= (key,)
        if intrinsics.isa(key, tuple):
            only_ints= True
            for i in key:
                if intrinsics.isa(i, int):
                    pass
                elif intrinsics.isa(i, slice):
                    only_ints=False
            if only_ints and len(key)==len(self.shape):

                intrinsics.call_builtin("array.set", None, [self]+list(key)+[_convert_to_dtype(value,self._dtype)])
            else:
                view = self[key]
                if intrinsics.isa(value,ndarray):
                    if len(view.shape)!=len(value.shape):
                        intrinsics.not_implemented()
                    intrinsics.call_builtin("array.copy", None, [view, value])
                else:
                    view_shape=intrinsics.typeof(view.shape)
                    #dtype=self._dtype
                    value=_convert_to_dtype(value,self._dtype)
                    l = intrinsics.bind(lambda indices: value, [view_shape])
                    intrinsics.call_builtin("array.fill", None, [view, l])
    @hipy.compiled_function
    def reshape(self, shape):
        if intrinsics.isa(shape, int):
            shape = (shape,)
        res = intrinsics.call_builtin("array.reshape", intrinsics.create_type(ndarray, self._dtype, shape), [self, shape])
        res.shape=shape
        return res


@hipy.classdef
class _concrete_ndarray(ndarray):
    def __init__(self, nested_list, dtype, shape,dims,shape_value=None):
        self._nested_list = nested_list
        self._dims=dims
        super().__init__(None, dtype, shape,shape_value=shape_value)
    @staticmethod
    @hipy.compiled_function
    def _compute_shape(l,ndims):
        curr=l
        r=[]
        for i in range(ndims):
            r.append(len(curr))
            if i <ndims-1:
                curr=curr[0]

        return r



    @staticmethod
    @hipy.raw
    def __create__(nested_list, _context):
        r = _infer_shape_dtype_of_nested_list(nested_list, _context)
        match r:
            case ValueHolder(value=hipy.lib.builtins.tuple(_elts=[_shape, _dtype])):
                match _shape:
                    case ValueHolder(value=hipy.lib.builtins.tuple(_elts=items)):
                        _shape=tuple([None for i in items])
        shape_value= _context.perform_call(_context.wrap(HLCFunctionValue(_concrete_ndarray._compute_shape)),[nested_list,_context.constant(len(_shape))])
        return _context.wrap(_concrete_ndarray(nested_list, _dtype, _shape,_context.constant(len(_shape)),shape_value))


    def __abstract__(self, context):
        abstract_list=self._nested_list.as_abstract(context)
        return context.perform_call(context.wrap(HLCFunctionValue(array)), [abstract_list])

    @hipy.compiled_function
    def __getitem__(self, o_item):
        item = o_item
        if intrinsics.isa(item, int):
            item= (item,)
        if intrinsics.isa(item, slice):
            item= (item,)
        if intrinsics.isa(item,tuple):
            only_ints= True
            for i in item:
                if intrinsics.isa(i, int):
                    pass
                elif intrinsics.isa(i, slice):
                    only_ints=False
            exactly_shape=len(item)==self._dims

            if only_ints and exactly_shape:
                r = self._nested_list
                for d in range(self._dims):
                    r=r[item[d]]
                return r
            else:
                return intrinsics.as_abstract(self)[o_item]

        else:
            intrinsics.not_implemented()





@hipy.compiled_function
def empty(shape, dtype=float64, order='C', like=None):
    intrinsics.only_implemented_if(order == 'C', like is None)
    if intrinsics.isa(shape, int):
        shape = (shape,)
    res=intrinsics.call_builtin("array.create_empty", intrinsics.create_type(ndarray, dtype, shape), [shape])
    res.shape=shape
    return res


@hipy.compiled_function
def ones(shape, dtype=float64, order='C', like=None):
    intrinsics.only_implemented_if(order == 'C', like is None)
    if intrinsics.isa(shape, int):
        shape = (shape,)
    res = empty(shape, dtype, order, like)
    value=_convert_to_dtype(1,dtype)
    fill_fn = intrinsics.bind(lambda indices: value, [intrinsics.typeof(shape)])

    intrinsics.call_builtin("array.fill", None, [res, fill_fn])
    return res
@hipy.compiled_function
def zeros(shape, dtype=float64, order='C', like=None):
    intrinsics.only_implemented_if(order == 'C', like is None)
    if intrinsics.isa(shape, int):
        shape = (shape,)
    res = empty(shape, dtype, order, like)
    value=_convert_to_dtype(0,dtype)
    fill_fn = intrinsics.bind(lambda indices: value, [intrinsics.typeof(shape)])

    intrinsics.call_builtin("array.fill", None, [res, fill_fn])
    return res

@hipy.compiled_function
def zeros_like(a):
    return zeros(a.shape, a._dtype)

@hipy.raw
def _infer_shape_dtype_of_nested_list(l,_context):
    def _infer_shape_dtype_of_nested_list_(t):
        if isinstance(t, hipy.lib.builtins.list.ListType):
            rshape, rdtype=_infer_shape_dtype_of_nested_list_(t.element_type)
            return 1+rshape, rdtype
        else:
            return 0, t
    if isinstance(l.value, hipy.lib.builtins.list):
         rshape, rdtype=_infer_shape_dtype_of_nested_list_(l.value.__hipy_get_type__())
         shape_tuple= _context.wrap(hipy.lib.builtins.tuple([_context.constant(None) for i in range(rshape)]))
         dtype= _context.wrap(TypeValue(rdtype))
         return _context.create_tuple([shape_tuple, dtype])
    else:
        raise NotImplementedError()


@hipy.raw
def _all_concrete_lists(l, _context):
    def internal(l, pos, shape):
        match l:
            case ValueHolder(value=hipy.lib.builtins._concrete_list(items=items)):
                if pos>=len(shape):
                    shape.append(None)
                if shape[pos] is not None and len(items) != shape[pos]:
                    return False
                shape[pos] = len(items)
                for i in items:
                    if not internal(i, pos + 1, shape):
                        return False
                return True
            case ValueHolder(value=hipy.lib.builtins.list()):
                return False
            case _:
                return True # todo: refine
    return _context.constant(internal(l, 0, []))


@hipy.compiled_function
def array(l):
    if intrinsics.isa(l, list):
        if _all_concrete_lists(l):
            return _concrete_ndarray(l)
        else:
            shape, dtype=_infer_shape_dtype_of_nested_list(l)
            convert_fn= intrinsics.bind(lambda v: _convert_to_np_type(v), [dtype])
            res=intrinsics.call_builtin("array.from_nested_list", intrinsics.create_type(ndarray, convert_fn.res_type, shape), [l,convert_fn])
            return res
    else:
        intrinsics.not_implemented()

@hipy.compiled_function
def isnan(x):
    if intrinsics.isa(x, float64):
        return intrinsics.call_builtin("scalar.float.isnan", bool, [x])
    elif intrinsics.isa(x, ndarray):
        return _array_apply_scalar(x, isnan)
    else:
        intrinsics.not_implemented()


import hipy.lib.numpy.random