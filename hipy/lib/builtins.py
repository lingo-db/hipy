import ast
import inspect
from typing import List, Tuple, Dict, Any

from builtins import *
import builtins
import hipy
from hipy.value import CValue, ValueHolder, Value, HLCClassValue, TypeValue, Type, SimpleType, static_object, RawValue, \
    AnyType, ConstIterValue, HLCFunctionValue
import hipy.ir as ir
import hipy.intrinsics as intrinsics

__HIPY_MODULE__ = "builtins"


@hipy.classdef
class object(Value):
    def __init__(self, value, known_object=None):
        super().__init__(value)
        self._known_object = known_object

    @hipy.compiled_function
    def __topython__(self):
        return self

    @hipy.compiled_function
    def __add__(self, other):
        return intrinsics.call_builtin("python.operator.add", object, [self, intrinsics.to_python(other)])

    @hipy.compiled_function
    def __sub__(self, other):
        return intrinsics.call_builtin("python.operator.sub", object, [self, intrinsics.to_python(other)])

    @hipy.compiled_function
    def __mul__(self, other):
        return intrinsics.call_builtin("python.operator.mul", object, [self, intrinsics.to_python(other)])

    @hipy.compiled_function
    def __truediv__(self, other):
        return intrinsics.call_builtin("python.operator.div", object, [self, intrinsics.to_python(other)])

    @hipy.compiled_function
    def __eq__(self, other):
        return intrinsics.call_builtin("python.operator.eq", object, [self, intrinsics.to_python(other)])

    @hipy.compiled_function
    def __ne__(self, other):
        return not (self == other)

    @hipy.compiled_function
    def __lt__(self, other):
        return intrinsics.call_builtin("python.operator.lt", object, [self, intrinsics.to_python(other)])

    @hipy.compiled_function
    def __gt__(self, other):
        return intrinsics.call_builtin("python.operator.gt", object, [self, intrinsics.to_python(other)])

    @hipy.compiled_function
    def __ge__(self, item):
        return not (self < item)

    @hipy.compiled_function
    def __le__(self, item):
        return not (self > item)

    @hipy.compiled_function
    def __pow__(self, power, modulo=None):
        if intrinsics.isa(power, _const_int):
            if power<5:
                res=self
                for i in range(1,power):
                    res=res*self
                return res
        return intrinsics.call_builtin("python.operator.pow", object, [self, power])

    @hipy.raw
    def __call__(self, *args, _context, **kwargs):
        def py_func_get_return_type(fn):
            try:
                fn_source = inspect.getsource(fn)
                fn_ast = ast.parse(fn_source)
                match fn_ast:
                    case ast.Module(body=[ast.FunctionDef(returns=ast.Name(id=return_type))]):
                        if return_type == "int":
                            return int.__hipy_create_type__(), "scalar.int.from_python"
                        elif return_type == "float":
                            return float.__hipy_create_type__(), "scalar.float.from_python"
                        elif return_type == "bool":
                            return bool.__hipy_create_type__(), "scalar.bool.from_python"
                        elif return_type == "str":
                            return str.__hipy_create_type__(), "scalar.string.from_python"
                    case _:
                        return None, None
            except:
                return None, None

        to_python = lambda v: _context.to_python(v)
        py_res = ValueHolder(object(
            ir.PythonCall(_context.block, self.get_ir_value(_context),
                          [to_python(a).get_ir_value(_context) for a in args],
                          [(n, to_python(a).get_ir_value(_context)) for n, a in kwargs.items()], ).result),
            _context)
        ret_type, conversion = py_func_get_return_type(self.value._known_object)
        if ret_type is not None:
            return _context.wrap(_context.call_builtin(conversion, ret_type, [py_res]))
        return py_res

    @hipy.raw
    def __hipy_setattr__(self, attr, value, _context):
        match attr:
            case ValueHolder(value=attr):
                pass
        match attr:
            case CValue(cval=attr):
                return ValueHolder(object(ir.PySetAttr(_context.block, attr, self.get_ir_value(_context),
                                                       _context.to_python(value).get_ir_value(_context)).result),
                                   _context)
            case _:
                raise NotImplementedError()

    @hipy.raw
    def __hipy_getattr__(self, item, _context):
        match item:
            case ValueHolder(value=item):
                pass
        match item:
            case CValue(cval=attr):
                return ValueHolder(object(ir.PyGetAttr(_context.block, attr, self.get_ir_value(_context)).result),
                                   _context)
            case _:
                raise NotImplementedError()

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        return self, other_fn(lambda c: c.to_python(other)), lambda val: object(val)

    class PythonObjectType(Type):
        def __init__(self, known_object=None):
            self.known_object = known_object

        def ir_type(self):
            return ir.pyobj

        def construct(self, value, context):
            return object(value, self.known_object)

        def __eq__(self, other):
            return isinstance(other, object.PythonObjectType)

    @staticmethod
    def __hipy_create_type__(*args) -> Type:
        return object.PythonObjectType(None)

    def __hipy_get_type__(self):
        return object.PythonObjectType(self._known_object)

    @hipy.compiled_function
    def __hipy__repr__(self):
        repr = self.__repr__()
        return intrinsics.call_builtin("scalar.string.from_python", str, [repr])


@hipy.classdef
class bool(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)

    @staticmethod
    @hipy.raw
    def __create__(val, _context):
        return _context._to_bool(val)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.call_builtin("scalar.bool.to_python", object, [self])

    @staticmethod
    def get_ir_type():
        return ir.bool

    @hipy.compiled_function
    def __str__(self):
        return intrinsics.call_builtin("scalar.bool.to_string", str, [self])

    @hipy.compiled_function
    def __and__(self, other):
        return self and other

    @hipy.compiled_function
    def __or__(self, other):
        return self or other

    @hipy.compiled_function
    def __invert__(self):
        return not self

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, bool):
            return self, other, lambda val: bool(val)
        else:
            raise NotImplementedError()

    @staticmethod
    def __hipy_create_type__(*args) -> Type:
        return SimpleType(bool, ir.bool)

    def __hipy_get_type__(self):
        return SimpleType(bool, ir.bool)

    @hipy.compiled_function
    def __int__(self):
        return 1 if self else 0


@hipy.classdef
class _const_bool(CValue, bool):
    def __init__(self, cval):
        bool.__init__(self, None)
        CValue.__init__(self, cval)

    def __abstract__(self, _context):
        return bool(ir.Constant(_context.block, self.cval, ir.bool).result)

    @hipy.compiled_function
    def __hipy__repr__(self):
        return str(self)


@hipy.classdef
class int(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)

    @staticmethod
    @hipy.raw
    def _cast_to_int(value, _context):
        try:
            with _context.no_fallback():
                return _context.perform_call(_context.get_attr(value, "__int__"))
        except (NotImplementedError, AttributeError) as e:
            raise NotImplementedError("Cannot cast to int") from e

    @staticmethod
    @hipy.compiled_function
    def __create__(val):
        if intrinsics.isa(val, int):
            return val
        elif intrinsics.isa(val, float):
            return intrinsics.call_builtin("scalar.float.to_int", int, [val])
        elif intrinsics.isa(val, str):
            return intrinsics.call_builtin("scalar.int.from_string", int, [val])
        else:
            return int._cast_to_int(val)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.call_builtin("scalar.int.to_python", object, [self])

    @hipy.compiled_function
    def __str__(self):
        return intrinsics.call_builtin("scalar.int.to_string", str, [self])

    @hipy.compiled_function
    def __bool__(self):
        return self != 0

    @hipy.compiled_function
    def _int_op(self, op, other, reverse=False):
        if intrinsics.isa(other, int):
            left = other if reverse else self
            right = self if reverse else other
            # todo: other width?
            return intrinsics.call_builtin("scalar.int." + op, int, [left, right])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def _cmp_op(self, op, other):
        if intrinsics.isa(other, int):
            return intrinsics.call_builtin("scalar.int.compare." + op, bool, [self, other])
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
    def __floordiv__(self, other):
        return self._int_op("div", other)

    @hipy.compiled_function
    def __rtruediv__(self, other):
        if intrinsics.isa(other, int):
            return float(other) / float(self)
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __mod__(self, other):
        return self._int_op("mod", other)

    @hipy.compiled_function
    def __rmod__(self, other):
        return self._int_op("mod", other, reverse=True)

    @hipy.compiled_function
    def __iadd__(self, other):
        return self + other

    @hipy.compiled_function
    def __isub__(self, other):
        return self - other

    @hipy.compiled_function
    def __imul__(self, other):
        return self * other

    @hipy.compiled_function
    def __itruediv__(self, other):
        return self / other

    @hipy.compiled_function
    def __imod__(self, other):
        return self % other

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, int):
            return self, other, lambda val: int(val)
        else:
            raise NotImplementedError()

    @staticmethod
    def __hipy_create_type__(*args) -> Type:
        return SimpleType(int, ir.int)

    def __hipy_get_type__(self):
        return SimpleType(int, ir.int)

    @hipy.compiled_function
    def __hipy__repr__(self):
        return str(self)


@hipy.classdef
class _const_int(CValue, int):
    __HIPY_MATERIALIZED__ = False

    def __init__(self, cval):
        int.__init__(self, None)
        CValue.__init__(self, cval)

    def __abstract__(self, _context):
        return int(ir.Constant(_context.block, self.cval, ir.int).result)


@hipy.classdef
class float(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)

    @staticmethod
    @hipy.raw
    def _cast_to_float(value, _context):
        try:
            with _context.no_fallback():
                return _context.perform_call(_context.get_attr(value, "__float__"))
        except (NotImplementedError, AttributeError) as e:
            raise NotImplementedError("Cannot cast to float") from e

    @staticmethod
    @hipy.compiled_function
    def __create__(val):
        if intrinsics.isa(val, float):
            return val
        elif intrinsics.isa(val, int):
            return intrinsics.call_builtin("scalar.float.from_int", float, [val])
        else:
            return float._cast_to_float(val)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.call_builtin("scalar.float.to_python", object, [self])

    @staticmethod
    def get_ir_type():
        return ir.f64

    @hipy.compiled_function
    def __str__(self):
        return intrinsics.call_builtin("scalar.float.to_string", str, [self])

    @hipy.compiled_function
    def _float_op(self, op, other, reverse=False):
        if intrinsics.isa(other, int):
            return self._float_op(op, float(other), reverse)
        elif intrinsics.isa(other, float):
            left = other if reverse else self
            right = self if reverse else other
            # todo: other width?
            return intrinsics.call_builtin("scalar.float." + op, float, [left, right])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def _cmp_op(self, op, other):
        if intrinsics.isa(other, int):
            return self._cmp_op(op, float(other))
        if intrinsics.isa(other, float):
            return intrinsics.call_builtin("scalar.float.compare." + op, bool, [self, other])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __neg__(self):
        return intrinsics.call_builtin("scalar.float.neg", float, [self])

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
    def __iadd__(self, other):
        return self + other

    @hipy.compiled_function
    def __isub__(self, other):
        return self - other

    @hipy.compiled_function
    def __imul__(self, other):
        return self * other

    @hipy.compiled_function
    def __itruediv__(self, other):
        return self / other

    @hipy.compiled_function
    def __pow__(self, other):
        return self._float_op("pow", other)

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, float):
            return self, other, lambda val: float(val)
        else:
            raise NotImplementedError()

    @staticmethod
    def __hipy_create_type__(*args) -> Type:
        return SimpleType(float, ir.f64)

    def __hipy_get_type__(self):
        return SimpleType(float, ir.f64)

    @hipy.compiled_function
    def __hipy__repr__(self):
        return str(self)


@hipy.classdef
class _const_float(CValue, float):
    def __init__(self, cval):
        float.__init__(self, None)
        CValue.__init__(self, cval)

    def __abstract__(self, _context):
        return float(ir.Constant(_context.block, self.cval, ir.f64).result)


@hipy.classdef
class str(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)

    @staticmethod
    @hipy.raw
    def __create__(value, _context):

        def object_to_str(value):
            res = _context.perform_call(_context.get_attr(_context.import_pymodule("builtins"), "str"), [value])
            return _context.wrap(
                _context.call_builtin("scalar.string.from_python", str.__hipy_create_type__(), [res]))

        match value:
            case ValueHolder(value=raw_value):
                pass
        match raw_value:
            case str():
                return value
            case CValue(cval=val):
                return _context.constant(builtins.str(val))
            case object():
                return object_to_str(value)
            case _:
                try:
                    with _context.no_fallback():
                        return _context.perform_call(_context.get_attr(value, "__str__"))
                except (NotImplementedError, AttributeError):
                    try:
                        with _context.no_fallback():
                            return _context.perform_call(_context.get_attr(value, "__hipy__repr__"))
                    except (NotImplementedError, AttributeError):
                        # todo: implement
                        return object_to_str(_context.to_python(value))

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.call_builtin("scalar.string.to_python", object, [self])

    @hipy.compiled_function
    def strip(self):
        return intrinsics.call_builtin("scalar.string.strip", str, [self])
    @hipy.compiled_function
    def rstrip(self):
        return intrinsics.call_builtin("scalar.string.rstrip", str, [self])

    @hipy.compiled_function
    def __iadd__(self, other):
        return self + other

    @hipy.compiled_function
    def __add__(self, other):
        if intrinsics.isa(other, str):
            return intrinsics.call_builtin("scalar.string.concatenate", str, [self, other])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def _cmp_op(self, op, other):
        if intrinsics.isa(other, str):
            return intrinsics.call_builtin("scalar.string.compare." + op, bool, [self, other])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __eq__(self, other):
        return self._cmp_op("eq", other)

    @hipy.compiled_function
    def __ne__(self, other):
        return not (self == other)

    @hipy.compiled_function
    def __lt__(self, other):
        return self._cmp_op("lt", other)

    @hipy.compiled_function
    def __le__(self, other):
        return self._cmp_op("lte", other)

    @hipy.compiled_function
    def __gt__(self, other):
        return not (self <= other)

    @hipy.compiled_function
    def __ge__(self, other):
        return not (self < other)

    @hipy.compiled_function
    def __contains__(self, item):
        if intrinsics.isa(item, str):
            return intrinsics.call_builtin("scalar.string.contains", bool, [self, item])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def lower(self):
        return intrinsics.call_builtin("scalar.string.lower", str, [self])

    @hipy.compiled_function
    def upper(self):
        return intrinsics.call_builtin("scalar.string.upper", str, [self])

    @hipy.compiled_function
    def find(self, sub, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self)
        return intrinsics.call_builtin("scalar.string.find", int, [self, sub, start, end])

    @hipy.compiled_function
    def rfind(self, sub, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self)
        return intrinsics.call_builtin("scalar.string.rfind", int, [self, sub, start, end])

    @hipy.compiled_function
    def partition(self, sep):
        pos = self.find(sep)
        if pos == -1:
            return self, "", ""
        else:
            return self[:pos], sep, self[pos + len(sep):]

    @hipy.compiled_function
    def rpartition(self, sep):
        pos = self.rfind(sep)
        if pos == -1:
            return "", "", self
        else:
            return self[:pos], sep, self[pos + len(sep):]

    @hipy.compiled_function
    def replace(self, old, new, count=None):
        if count is None:
            return intrinsics.call_builtin("scalar.string.replace", str, [self, old, new])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __getitem__(self, item):
        if intrinsics.isa(item, int):
            return intrinsics.call_builtin("scalar.string.at", str, [self, item])
        elif intrinsics.isa(item, slice):
            length = len(self)
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else length
            start = start if start >= 0 else length + start
            stop = stop if stop >= 0 else length + stop
            if item.step is None:
                return intrinsics.call_builtin("scalar.string.substr", str, [self, start, stop])
            else:
                return "".join([self[i] for i in range(start, stop, item.step)])
        else:
            intrinsics.not_implemented()

    @staticmethod
    def __hipy_create_type__(*args) -> Type:
        return SimpleType(str, ir.string)

    def __hipy_get_type__(self):
        return SimpleType(str, ir.string)

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, str):
            return self, other, lambda val: str(val)
        else:
            raise NotImplementedError()

    @hipy.compiled_function
    def join(self, iter):
        res = ""
        first = True
        for v in iter:
            if not first:
                res = res + self
            first = False
            res = res + v
        return res

    @hipy.compiled_function
    def __hipy__repr__(self):
        return "'" + self + "'"

    @hipy.compiled_function
    def __len__(self):
        return intrinsics.call_builtin("scalar.string.length", int, [self])

    @hipy.compiled_function
    def split(self, pattern, maxsplit=-1):
        return intrinsics.call_builtin("scalar.string.split", intrinsics.create_type(list, str),
                                       [self, pattern, maxsplit])

    @hipy.classdef
    class _iterator(Value):
        def __init__(self, str_val, value=None):
            super().__init__(value)
            self._str_val = str_val

        def __track__(self, iter_value, context):
            pass

        @hipy.compiled_function
        def __itertype__(self):
            return str

        @hipy.compiled_function
        def __iterate__(self, loopfn, x, iter_vals):
            return intrinsics.call_builtin("scalar.string.iter", intrinsics.typeof(iter_vals),
                                           [loopfn, x, iter_vals, self._str_val])

        @hipy.compiled_function
        def __topython__(self):
            return self._str_val.__topython__().__iter__()

        def __abstract__(self, context):
            self._str_val = self._str_val.as_abstract(context)
            return str._iterator(self._str_val, self._str_val.value.__value__)

        class T(Type):

            def ir_type(self):
                return str

            def construct(self, value, context):
                return str._iterator(SimpleType(str, ir.string).construct(value, context), value)

            def __eq__(self, other):
                return isinstance(other, str._iterator.T)

        @staticmethod
        def __hipy_create_type__() -> Type:
            return str._iterator.T()

        def __hipy_get_type__(self) -> Type:
            return str._iterator.T()

    @hipy.raw
    def __iter__(self, _context):
        return _context.wrap(str._iterator(self.as_abstract(_context)))

    @hipy.compiled_function
    def isdigit(self):
        return ord('0') <= ord(self) <= ord('9')

    @hipy.compiled_function
    def isascii(self):
        res = True
        for c in self:
            if ord(c) > 127:
                res = False
        return res


@hipy.classdef
class _const_str(CValue, str):
    __HIPY_MATERIALIZED__ = False

    def __init__(self, cval):
        str.__init__(self, None)
        CValue.__init__(self, cval)

    def __abstract__(self, _context):
        return str(ir.Constant(_context.block, self.cval, ir.string).result)


@hipy.classdef
class list(Value):
    def __init__(self, value, element_type):
        super().__init__(value)
        self._element_type = element_type

    @staticmethod
    @hipy.raw
    def __create__(value, _context):
        match value.value:
            case list():
                return value
            case object():
                raise NotImplementedError()
            case _:
                with _context.no_fallback():
                    @hipy.compiled_function
                    def from_iterable(value):
                        return [v for v in value]

                    return _context.perform_call(_context.get_by_name(from_iterable, 'from_iterable'), [value])
        raise NotImplementedError()

    @hipy.compiled_function
    def __topython__(self):
        l = intrinsics.call_builtin("python.create_list", object, [])
        for item in self:
            l.append(item)
        # intrinsics.not_implemented()
        return l

    class ListType(Type):
        def __init__(self, element_type: Type):
            self.element_type = element_type

        def ir_type(self):
            return ir.ListType(self.element_type.ir_type())

        def construct(self, value, context):
            return list(value, self.element_type)

        def __eq__(self, other):
            if isinstance(other, list.ListType):
                return self.element_type == other.element_type
            else:
                return False

        def __repr__(self):
            return f"list.ListType({self.element_type})"

        def get_cls(self):
            return list

    @staticmethod
    def __hipy_create_type__(*args) -> Type:
        t=args[0]
        if isinstance(t, builtins.list):
            t=t[0]
        return list.ListType(t)

    def __hipy_get_type__(self):
        return list.ListType(self._element_type)

    @hipy.compiled_function
    def append(self, item):
        intrinsics.try_narrow(item, self._element_type)
        if intrinsics.isa(item, self._element_type):
            intrinsics.track_nested(item, self)
            intrinsics.call_builtin("list.append", None, [self, item])
        else:
            intrinsics.not_implemented()
    @hipy.compiled_function
    def sort(self):
        compare_fn = intrinsics.bind(lambda l, r: l < r, [self._element_type, self._element_type])
        intrinsics.call_builtin("list.sort", None, [self, compare_fn])

    @hipy.compiled_function
    def __lt__(self, other):
        if intrinsics.isa(other, list):
            if self._element_type == other._element_type:
                is_lt = False
                is_gt = False
                for i in range(0, min(len(self), len(other))):
                    l = self[i]
                    r = other[i]
                    if l < r:
                        is_lt = True
                    if l > r:
                        is_gt = True
                return is_lt and not is_gt
            else:
                intrinsics.not_implemented()
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __add__(self, other):
        if intrinsics.isa(other, list):
            if self._element_type == other._element_type:
                res = []
                for item in self:
                    res.append(item)
                for item in other:
                    res.append(item)
                return res
            else:
                intrinsics.not_implemented()
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __mul__(self, multiplier):
        if intrinsics.isa(multiplier, int):
            res = []
            for i in range(multiplier):
                for item in self:
                    res.append(item)

            return res
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __setitem__(self, key, value):
        if intrinsics.isa(key, int) and intrinsics.isa(value, self._element_type):
            intrinsics.call_builtin("list.set", None, [self, key, value])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __getitem__(self, item):
        if intrinsics.isa(item, int):
            return intrinsics.call_builtin("list.at", self._element_type, [self, item])
        elif intrinsics.isa(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else len(self)
            step = item.step if item.step is not None else 1
            return [self[i] for i in range(start, stop, step)]
        else:
            intrinsics.not_implemented()

    @hipy.classdef
    class _iterator(Value):
        def __init__(self, list_val, value=None):
            super().__init__(value)
            self._list_val = list_val

        def __track__(self, iter_value, context):
            context.track_nested(iter_value, self._list_val)

        @hipy.compiled_function
        def __itertype__(self):
            return self._list_val._element_type

        @hipy.compiled_function
        def __iterate__(self, loopfn, x, iter_vals):
            return intrinsics.call_builtin("list.iter", intrinsics.typeof(iter_vals),
                                           [loopfn, x, iter_vals, self._list_val])

        @hipy.compiled_function
        def __topython__(self):
            return intrinsics.to_python(self._list_val).__iter__()

        def __abstract__(self, context):
            self._list_val = self._list_val.as_abstract(context)
            return list._iterator(self._list_val, self._list_val.value.__value__)

        class T(Type):
            def __init__(self, list_type):
                self.list_type = list_type

            def ir_type(self):
                return self.list_type.ir_type()

            def construct(self, value, context):
                return list._iterator(self.list_type.construct(value, context), value)

            def __eq__(self, other):
                if isinstance(other, list._iterator.T):
                    return self.list_type == other.list_type
                else:
                    return False

        @staticmethod
        def __hipy_create_type__(*args) -> Type:
            return list._iterator.T(args[0])

        def __hipy_get_type__(self) -> Type:
            return list._iterator.T(self._list_val.__hipy_get_type__())

    @hipy.raw
    def __iter__(self, _context):
        return _context.wrap(list._iterator(self.as_abstract(_context)))

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        def to_empty_list(element_type):
            def fn(context):
                return context.wrap(context.call_builtin("list.create", list.ListType(element_type), []))

            return fn

        if isinstance(other.value, list):
            if self.value._element_type == other.value._element_type:
                return self, other, lambda val: list(val, self.value._element_type)
            elif isinstance(self.value._element_type, AnyType):
                return self_fn(to_empty_list(other.value._element_type)), other, lambda val: list(val,
                                                                                                  other.value._element_type)
            elif isinstance(other.value._element_type, AnyType):
                return self, other_fn(to_empty_list(self.value._element_type)), lambda val: list(val,
                                                                                                 self.value._element_type)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    @hipy.compiled_function
    def __hipy__repr__(self):
        return "[" + ", ".join([repr(v) for v in self]) + "]"

    @hipy.compiled_function
    def __len__(self):
        return intrinsics.call_builtin("list.length", int, [self])

    @hipy.compiled_function
    def __contains__(self, item):
        found = False
        for i in self:
            if i == item:
                found = True
        return found


def _common_type(types):
    same_type = None
    for t in types:
        if same_type is None:
            same_type = t
        elif same_type != t:
            return object.PythonObjectType()
    if same_type is None:
        return AnyType()
    return same_type


@hipy.classdef
class _concrete_list(list):
    def __init__(self, items, element_type=None):
        super().__init__(None, element_type)
        self.items = items
        self.update_type()

    def update_type(self):
        if len(self.items) == 0 and self._element_type is not None:
            return
        self._element_type = _common_type([item.value.__hipy_get_type__() for item in self.items])

    def __abstract__(self, _context):
        l = ValueHolder(_context.call_builtin("list.create", list.ListType(self._element_type), []), _context)

        for item in self.items:
            match item:
                case ValueHolder(value=value):
                    if self._element_type == object.PythonObjectType() and value.__hipy_get_type__() != self._element_type:
                        item = _context.to_python(item)
                    _context.perform_call(_context.get_attr(l, "append"), [item])
                case _:
                    raise NotImplementedError()
        return l

    @hipy.raw
    def __getitem__(self, item, _context):
        match item:
            case ValueHolder(value=CValue(cval=item)):
                return self.value.items[item]
            case _:
                return _context.perform_call(_context.get_attr(self.as_abstract(_context), "__getitem__"),
                                             [item])

    @hipy.raw
    def append(self, item):
        self.value = _concrete_list(self.value.items + [item])

    @hipy.raw
    def __add__(self, other, _context):
        if isinstance(other.value, _concrete_list):
            return _context.wrap(_concrete_list(self.value.items + other.value.items))
        elif isinstance(other.value, list):
            return _context.perform_call(_context.get_attr(self.as_abstract(_context), "__add__"),
                                         [other])
        else:
            raise NotImplementedError()

    def __hipy_get_type__(self):
        return list.ListType(self._element_type)

    @hipy.raw
    def __constiter__(self, _context):
        return _context.wrap(ConstIterValue(self, self.value.items))

    @hipy.raw
    def __len__(self, _context):
        return _context.constant(len(self.value.items))

    @staticmethod
    def __narrow_type__(value, type):

        list_val: _concrete_list = value.value
        match type:
            case list.ListType(element_type=element_type):
                if list_val._element_type == AnyType():
                    value.value = _concrete_list(list_val.items, element_type)
                else:
                    raise NotImplementedError()
            case _:
                raise NotImplementedError()


@hipy.classdef
class dict(Value):
    def __init__(self, value, key_type, value_type):
        super().__init__(value)
        self._key_type = key_type
        self._value_type = value_type

    @hipy.compiled_function
    def __topython__(self):
        l = intrinsics.call_builtin("python.create_dict", object, [])
        for k in self:
            l[k] = self[k]
        return l

    class DictType(Type):
        def __init__(self, key_type: Type, value_type: Type):
            self.key_type = key_type
            self.value_type = value_type

        def ir_type(self):
            return ir.DictType(self.key_type.ir_type(), self.value_type.ir_type())

        def construct(self, value, context):
            return dict(value, self.key_type, self.value_type)

        def __eq__(self, other):
            if isinstance(other, dict.DictType):
                return self.key_type == other.key_type and self.value_type == other.value_type
            else:
                return False

    @staticmethod
    def __hipy_create_type__(*args) -> Type:
        return dict.DictType(args)

    def __hipy_get_type__(self):
        return dict.DictType(self._key_type, self._value_type)

    @hipy.compiled_function
    def __getitem__(self, item):
        return intrinsics.call_builtin("dict.get", self._value_type, [self, item])

    @hipy.compiled_function
    def __setitem__(self, item, value):
        intrinsics.try_narrow(item, self._key_type)
        intrinsics.try_narrow(value, self._value_type)
        if intrinsics.isa(item, self._key_type) and intrinsics.isa(value, self._value_type):
            return intrinsics.call_builtin("dict.set", None, [self, item, value])
        else:
            intrinsics.not_implemented()

    @hipy.compiled_function
    def __hipy__repr__(self):
        return "{" + ", ".join([repr(k) + ": " + repr(self[k]) for k in self]) + "}"
    @hipy.compiled_function
    def __len__(self):
        return intrinsics.call_builtin("dict.length", int, [self])
    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        def to_empty_dict(key_type, value_type):
            def fn(context):
                k_type = context.wrap(TypeValue(key_type))
                create_cmp_fn_val = context.wrap(HLCFunctionValue(_create_cmp_fn))
                eq_fn = context.perform_call(create_cmp_fn_val, [k_type])
                return context.wrap(context.call_builtin("dict.create", dict.DictType(key_type, value_type), [eq_fn]))

            return fn

        if isinstance(other.value, dict):
            if self.value._key_type == other.value._key_type and self.value._value_type == other.value._value_type:
                return self, other, lambda val: dict(val, self.value._key_type, self.value._value_type)
            elif isinstance(self.value._key_type, AnyType) and isinstance(self.value._value_type, AnyType):
                return self_fn(to_empty_dict(other.value._key_type, other.value._value_type)), other, lambda val: dict(
                    val, other.value._key_type, other.value._value_type)
            elif isinstance(other.value._element_type, AnyType):
                return self, other_fn(to_empty_dict(self.value._key_type, self.value._value_type)), lambda val: dict(
                    val, self.value._key_type, self.value._value_type)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    @hipy.compiled_function
    def __contains__(self, item):
        return intrinsics.call_builtin("dict.contains", bool, [self, item])

    @hipy.compiled_function
    def setdefault(self, key, default):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    @hipy.compiled_function
    def get(self, key, default):
        if key in self:
            return self[key]
        else:
            return default

    @hipy.classdef
    class _iterator(Value):
        def __init__(self, dict_val, value=None):
            super().__init__(value)
            self._dict_val = dict_val

        def __track__(self, iter_value, context):
            context.track_nested(iter_value, self._dict_val)

        @hipy.compiled_function
        def __itertype__(self):
            return self._dict_val._key_type

        @hipy.compiled_function
        def __iterate__(self, loopfn, x, iter_vals):
            return intrinsics.call_builtin("dict.iter_keys", intrinsics.typeof(iter_vals),
                                           [loopfn, x, iter_vals, self._dict_val])

        @hipy.compiled_function
        def __topython__(self):
            return intrinsics.to_python(self._dict_val).__iter__()

        def __abstract__(self, context):
            self._dict_val = self._dict_val.as_abstract(context)
            return dict._iterator(self._dict_val, self._dict_val.value.__value__)

        class T(Type):
            def __init__(self, dict_type):
                self.dict_type = dict_type

            def ir_type(self):
                return self.dict_type.ir_type()

            def construct(self, value, context):
                return dict._iterator(self.dict_type.construct(value, context), value)

            def __eq__(self, other):
                if isinstance(other, dict._iterator.T):
                    return self.dict_type == other.dict_type
                else:
                    return False

        @staticmethod
        def __hipy_create_type__(*args) -> Type:
            return dict._iterator.T(args[0])

        def __hipy_get_type__(self) -> Type:
            return dict._iterator.T(self._dict_val.__hipy_get_type__())
    @hipy.classdef
    class _items(Value):
        def __init__(self, dict_val, value=None):
            super().__init__(value)
            self._dict_val = dict_val

        @hipy.raw
        def __iter__(self,_context):
            return _context.wrap(dict._items._iterator(self.value._dict_val))

        @hipy.compiled_function
        def __topython__(self):
            return intrinsics.to_python(self._dict_val).items()

        def __abstract__(self, context):
            self._dict_val = self._dict_val.as_abstract(context)
            return dict._items(self._dict_val, self._dict_val.value.__value__)

        class T(Type):
            def __init__(self, dict_type):
                self.dict_type = dict_type

            def ir_type(self):
                return self.dict_type.ir_type()

            def construct(self, value, context):
                return dict._items(self.dict_type.construct(value, context), value)

            def __eq__(self, other):
                if isinstance(other, dict._iterator.T):
                    return self.dict_type == other.dict_type
                else:
                    return False

        @staticmethod
        def __hipy_create_type__(*args) -> Type:
            return dict._items.T(args[0])

        def __hipy_get_type__(self) -> Type:
            return dict._items.T(self._dict_val.__hipy_get_type__())
        @hipy.classdef
        class _iterator(Value):
            def __init__(self, dict_val, value=None):
                super().__init__(value)
                self._dict_val = dict_val

            def __track__(self, iter_value, context):
                context.track_nested(iter_value, self._dict_val)

            @hipy.compiled_function
            def __itertype__(self):
                return intrinsics.create_type(tuple,[self._dict_val._key_type, self._dict_val._value_type])

            @hipy.compiled_function
            def __iterate__(self, loopfn, x, iter_vals):
                return intrinsics.call_builtin("dict.iter_items", intrinsics.typeof(iter_vals),
                                               [loopfn, x, iter_vals, self._dict_val])

            @hipy.compiled_function
            def __topython__(self):
                return intrinsics.to_python(self._dict_val).items().__iter__()

            def __abstract__(self, context):
                self._dict_val = self._dict_val.as_abstract(context)
                return dict._items._iterator(self._dict_val, self._dict_val.value.__value__)

            class T(Type):
                def __init__(self, dict_type):
                    self.dict_type = dict_type

                def ir_type(self):
                    return self.dict_type.ir_type()

                def construct(self, value, context):
                    return dict._items._iterator(self.dict_type.construct(value, context), value)

                def __eq__(self, other):
                    if isinstance(other, dict._iterator.T):
                        return self.dict_type == other.dict_type
                    else:
                        return False

            @staticmethod
            def __hipy_create_type__(*args) -> Type:
                return dict._items._iterator.T(args[0])

            def __hipy_get_type__(self) -> Type:
                return dict._items._iterator.T(self._dict_val.__hipy_get_type__())

    @hipy.raw
    def __iter__(self, _context):
        return _context.wrap(dict._iterator(self.as_abstract(_context)))

    @hipy.raw
    def items(self, _context):
        return _context.wrap(dict._items(self.as_abstract(_context)))

def _type_of_constant(cval):
    match cval:
        case builtins.bool():
            return SimpleType(bool, ir.bool)
        case builtins.int():
            return SimpleType(int, ir.i64)
        case builtins.float():
            return SimpleType(float, ir.f64)
        case builtins.str():
            return SimpleType(str, ir.string)

        case _:
            return object.PythonObjectType()

@hipy.compiled_function
def _create_cmp_fn(key_type):
    return intrinsics.bind(lambda l, r : l==r, [key_type, key_type])
@hipy.classdef
class _concrete_dict(dict):

    def __init__(self, c_dict: Dict[Any, ValueHolder], to_insert=None):

        super().__init__(None, None, None)
        self.c_dict = c_dict
        self._to_insert = to_insert if to_insert is not None else []
        self.update_types()

    def update_types(self):
        self._key_type = _common_type(
            [_type_of_constant(k) for k in self.c_dict.keys()] + [k.value.__hipy_get_type__() for k, v in
                                                                  self._to_insert])
        self._value_type = _common_type(
            [v.value.__hipy_get_type__() for v in self.c_dict.values()] + [v.value.__hipy_get_type__() for k, v in
                                                                            self._to_insert])

    def __abstract__(self, _context):
        dict_type=self.__hipy_get_type__()
        key_type=_context.wrap(TypeValue(self._key_type))
        create_cmp_fn_val = _context.wrap(HLCFunctionValue(_create_cmp_fn))
        eq_fn = _context.perform_call(create_cmp_fn_val, [key_type])
        l = ValueHolder(_context.call_builtin("dict.create", dict_type, [eq_fn]), _context)

        for k, v in self.c_dict.items():
            key = _context.constant(k)
            if self._key_type == object.PythonObjectType() and key.value.__hipy_get_type__() != self._key_type:
                key = _context.to_python(key)

            if self._value_type == object.PythonObjectType() and v.value.__hipy_get_type__() != self._value_type:
                v = _context.to_python(v)
            _context.perform_call(_context.get_attr(l, "__setitem__"), [key, v])
        for key, v in self._to_insert:
            if self._key_type == object.PythonObjectType() and key.value.__hipy_get_type__() != self._key_type:
                key = _context.to_python(key)

            if self._value_type == object.PythonObjectType() and v.__hipy_get_type__() != self._value_type:
                v = _context.to_python(v)
            _context.perform_call(_context.get_attr(l, "__setitem__"), [key, v])
        return l

    @hipy.raw
    def __getitem__(self, item, _context):
        if len(self.value._to_insert) > 0:
            return _context.perform_call(_context.get_attr(self.as_abstract(_context), "__getitem__"),
                                         [item])
        match item:
            case ValueHolder(value=CValue(cval=item)):
                return self.value.c_dict[item]
            case _:
                return _context.perform_call(_context.get_attr(self.as_abstract(_context), "__getitem__"),
                                             [item])

    @hipy.raw
    def __setitem__(self, item, value, _context):
        if len(self.value._to_insert) > 0:
            return _context.perform_call(_context.get_attr(self.as_abstract(_context), "__setitem__"),
                                         [item, value])
        match item:
            case ValueHolder(value=CValue(cval=item)):
                new_c_dict = {**self.value.c_dict}
                new_c_dict[item] = value
                self.value = _concrete_dict(new_c_dict)
            case _:
                self.value = _concrete_dict({**self.value.c_dict}, [(item, value)])

    @hipy.raw
    def __topython__(self, _context):
        dict_value = _context.pyobj(ir.CallBuiltin(_context.block, "python.create_dict", [], ir.pyobj).result)

        for key, value in self.value.c_dict.items():
            _context.perform_call(_context.get_attr(dict_value, "__setitem__"),
                                  [_context.constant(key), value])
        for key, value in self.value._to_insert:
            _context.perform_call(_context.get_attr(dict_value, "__setitem__"),
                                  [key, value])
        return dict_value

    @hipy.raw
    def __constiter__(self, _context):
        return _context.wrap(ConstIterValue(self, [_context.constant(i) for i in self.value.c_dict] + [k for k, v in
                                                                                                       self.value._to_insert]))

    @hipy.raw
    def __contains__(self, item, _context):
        if len(self.value._to_insert) > 0:
            return _context.perform_call(_context.get_attr(self.as_abstract(_context), "__contains__"),
                                         [item])
        if len(self.value.c_dict) == 0:
            return _context.constant(False)
        match item:
            case ValueHolder(value=CValue(cval=item)):
                return _context.constant(item in self.value.c_dict)
            case _:
                _context.perform_call(_context.get_attr(self.as_abstract(_context), "__contains__"),
                                      [item])

    def __hipy_get_type__(self):
        self.update_types()
        return dict.DictType(self._key_type, self._value_type)


@hipy.classdef
class range(static_object["start", "stop", "step"]):
    def __init__(self, start, stop, step):
        super().__init__(lambda args: range(*args), start, stop, step)

    @staticmethod
    @hipy.raw
    def __create__(*args, _context=None):
        match len(args):
            case 0:
                raise TypeError("range expected 1 arguments, got 0")
            case 1:
                start = _context.constant(0)
                stop = args[0]
                step = _context.constant(1)
            case 2:
                start = args[0]
                stop = args[1]
                step = _context.constant(1)

            case 3:
                start = args[0]
                stop = args[1]
                step = args[2]
            case _:
                raise TypeError("range expected at most 3 arguments, got " + str(builtins.len(args)))
        return hipy.value.ValueHolder(range(start, stop, step), _context)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.import_pymodule("__main__").__builtins__.range(self.start, self.stop, self.step)

    @hipy.classdef
    class _iterator(Value):
        def __init__(self, range_val, value=None):
            super().__init__(value)
            self._range_val = range_val

        def __track__(self, iter_value, context):
            pass

        @hipy.compiled_function
        def __itertype__(self):
            return int

        @hipy.compiled_function
        def __iterate__(self, loopfn, x, iter_vals):
            return intrinsics.call_builtin("range.iter", intrinsics.typeof(iter_vals),
                                           [loopfn, x, iter_vals, self._range_val.start, self._range_val.stop,
                                            self._range_val.step])

        @hipy.compiled_function
        def __topython__(self):
            return self._range_val.__topython__().__iter__()

        def __abstract__(self, context):
            self._range_val = self._range_val.as_abstract(context)
            return range._iterator(self._range_val, self._range_val.value.__value__)

        class T(Type):
            def __init__(self, range_type):
                self.range_type = range_type

            def ir_type(self):
                return self.range_type.ir_type()

            def construct(self, value, context):
                return range._iterator(self.range_type.construct(value, context), value)

            def __eq__(self, other):
                if isinstance(other, range._iterator.T):
                    return self.range_type == other.range_type
                else:
                    return False

        @staticmethod
        def __hipy_create_type__(*args) -> Type:
            return range._iterator.T(args[0])

        def __hipy_get_type__(self) -> Type:
            return range._iterator.T(self._range_val.__hipy_get_type__())

    @hipy.raw
    def __iter__(self, _context):
        return _context.wrap(range._iterator(self))

    @hipy.raw
    def __constiter__(self, _context):
        match self.value.start.value, self.value.stop.value, self.value.step.value:
            case CValue(cval=start), CValue(cval=stop), CValue(cval=step):
                r = builtins.range(start, stop, step)
                if len(r) <= 10:
                    return _context.wrap(ConstIterValue(self, [_context.constant(i) for i in r]))
        raise NotImplementedError()

@hipy.classdef
class enumerate(static_object["iterable",]):
    def __init__(self, iterable):
        super().__init__(lambda args: enumerate(*args), iterable)
    @staticmethod
    @hipy.raw
    def _construct(iterable, _context=None):
        return hipy.value.ValueHolder(enumerate(iterable), _context)
    @staticmethod
    @hipy.compiled_function
    def __create__(iterable):
        return enumerate._construct(iterable.__iter__())

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.import_pymodule("__main__").__builtins__.enumerate(self.iterable)


    @hipy.compiled_function
    def __itertype__(self):
        return intrinsics.create_type(tuple,[int,self.iterable.__itertype__()])

    @hipy.compiled_function
    def __iterate__(self, loopfn, x, iter_vals):
        #todo: maybe we should use a different type for the mutable counter...
        counter=[0]
        def wrapperfn(x, iter_vals,i):
            new_res=intrinsics.call_indirect(loopfn,[x,iter_vals,(counter[0],i)],intrinsics.typeof(iter_vals))
            counter[0]+=1
            return new_res

        wrapped_fn=intrinsics.bind(wrapperfn, [intrinsics.typeof(x), intrinsics.typeof(iter_vals), self.iterable.__itertype__()] )
        return self.iterable.__iterate__(wrapped_fn, x, iter_vals)

    def __track__(self, iter_value, context):
        pass

    @hipy.compiled_function
    def __iter__(self):
        return self


@hipy.classdef
class slice(static_object["start", "stop", "step"]):
    def __init__(self, start, stop, step):
        super().__init__(lambda args: slice(*args), start, stop, step)

    @staticmethod
    @hipy.raw
    def __create__(*args, _context=None):
        match len(args):
            case 0:
                raise TypeError("slice expected 1 arguments, got 0")
            case 1:
                start = _context.constant(None)
                stop = args[0]
                step = _context.constant(None)
            case 2:
                start = args[0]
                stop = args[1]
                step = _context.constant(None)

            case 3:
                start = args[0]
                stop = args[1]
                step = args[2]
            case _:
                raise TypeError("slice expected at most 3 arguments, got " + str(builtins.len(args)))
        return hipy.value.ValueHolder(slice(start, stop, step), _context)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.import_pymodule("__main__").__builtins__.slice(self.start, self.stop, self.step)

    @hipy.compiled_function
    def __hipy__repr__(self):
        return f"slice({self.start}, {self.stop}, {self.step})"


@hipy.classdef
class tuple(Value):
    __HIPY_MUTABLE__ = False

    def __init__(self, elts, value=None):
        super().__init__(value)
        self._elts = elts
        self._element_types = [elt.value.__hipy_get_type__() for elt in elts]

    class TupleType(Type):
        def __init__(self, element_types: List[Type]):
            self.element_types = element_types

        def ir_type(self):
            return ir.RecordType([(f"_elt{i}", t.ir_type()) for i, t in builtins.enumerate(self.element_types)])

        def construct(self, value, context):
            args = []
            for i, t in builtins.enumerate(self.element_types):
                n = f"_elt{i}"
                args.append(
                    context.wrap(t.construct(ir.RecordGet(context.block, t.ir_type(), value, n).result, context)))
            return tuple(args, value=value)

        def __eq__(self, other):
            if isinstance(other, tuple.TupleType):
                return self.element_types == other.element_types
            else:
                return False

    @staticmethod
    @hipy.raw
    def _from_const_list(l, _context):
        match l:
            case ValueHolder(value=_concrete_list(items=elts)):
                return _context.wrap(tuple(elts))
            case _:
                raise NotImplementedError()

    def __hipy_get_type__(self):
        return tuple.TupleType(self._element_types)

    def __hipy_create_type__(*args) -> Type:
        return tuple.TupleType(args[0])

    def __abstract__(self, _context):
        if self.__value__ is not None:
            return self.__value__
        else:
            curr_type = self.__hipy_get_type__()
            record = _context.make_record([f"_elt{i}" for i in builtins.range(len(self._elts))], self._elts)
            return _context.wrap(curr_type.construct(record, _context))

    @hipy.compiled_function
    def _elementwise_comparison(self, other, lt, gt, idx=0):
        if idx == len(self):
            return False
        elif lt(self[idx], other[idx]):
            return True
        elif gt(self[idx], other[idx]):
            return False
        else:
            return self._elementwise_comparison(other, lt, gt, idx + 1)

    @hipy.compiled_function
    def __lt__(self, other):
        return self._elementwise_comparison(other, lambda a, b: a < b, lambda a, b: a > b)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.call_builtin("python.tuple_from_list", object, [intrinsics.to_python(list(self))])

    @hipy.raw
    def __getitem__(self, item, _context):
        match item:
            case ValueHolder(value=CValue(cval=item)):
                return self.value._elts[item]
            case _:
                raise NotImplementedError()

    @hipy.raw
    def __constiter__(self, _context):
        return _context.wrap(ConstIterValue(self, self.value._elts))

    @hipy.compiled_function
    def __hipy__repr__(self):
        return "(" + ", ".join([repr(v) for v in self]) + ("," if len(self) == 1 else "") + ")"

    @hipy.raw
    def __len__(self, _context):
        return _context.constant(len(self.value._elts))

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, tuple) and self.value._element_types == other.value._element_types:
            return self, other, lambda val: self.value.__hipy_get_type__().construct(val, context)
        else:
            raise NotImplementedError()


@hipy.compiled_function
def print(*args):
    to_print = " ".join([str(arg) for arg in args])
    intrinsics.call_builtin("dbg.print", None, [str(to_print)])


@hipy.raw
def repr(value, _context):
    try:
        with _context.no_fallback():
            return _context.perform_call(_context.get_attr(value, "__hipy__repr__"))
    except (NotImplementedError, AttributeError):
        return _context.constant(builtins.repr(value.value))


@hipy.compiled_function
def len(val):
    return val.__len__()


@hipy.classdef
class bytes(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)

    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.call_builtin("scalar.bytes.to_python", object, [self])

    @staticmethod
    def __hipy_create_type__(*args) -> Type:
        return SimpleType(bytes, ir.string)

    def __hipy_get_type__(self):
        return SimpleType(bytes, ir.string)

    @hipy.compiled_function
    def __hipy__repr__(self):
        return repr(intrinsics.to_python(self))

    @hipy.compiled_function
    def __len__(self):
        return intrinsics.call_builtin("scalar.string.length", int, [self])

    @hipy.compiled_function
    def __getitem__(self, item):
        if intrinsics.isa(item, int):
            return intrinsics.call_builtin("scalar.string.at", bytes, [self, item])
        elif intrinsics.isa(item, slice):
            length = len(self)
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else length
            start = start if start >= 0 else length + start
            stop = stop if stop >= 0 else length + stop
            if item.step is None:
                return intrinsics.call_builtin("scalar.string.substr", bytes, [self, start, stop])
            else:
                return "".join([self[i] for i in range(start, stop, item.step)])
        else:
            intrinsics.not_implemented()



@hipy.classdef
class _const_bytes(CValue, bytes):
    def __init__(self, cval):
        bytes.__init__(self, None)
        CValue.__init__(self, cval)

    def __abstract__(self, _context):
        return bytes(ir.Constant(_context.block, self.cval.decode('ascii'), ir.string).result)


@hipy.compiled_function
def sum(l):
    r = 0
    if intrinsics.isa(l, list):
        if l._element_type == float:
            r = 0.0
    for i in l:
        r += i
    return r


@hipy.compiled_function
def min(*args):
    if len(args) == 1:
        l = args[0]
    else:
        l = args
    res = l[0]
    for i in l:
        res = res if res < i else i
    return res

@hipy.compiled_function
def max(*args):
    if len(args) == 1:
        l = args[0]
    else:
        l = args
    res = l[0]
    for i in l:
        res = res if res > i else i
    return res


@hipy.raw
def _const_ord(cval, _context):
    return _context.constant(builtins.ord(cval.value.cval))


@hipy.compiled_function
def ord(c):
    if intrinsics.isa(c, _const_str):
        return _const_ord(c)
    elif intrinsics.isa(c, str):
        return intrinsics.call_builtin("scalar.string.ord", int, [c])
    else:
        intrinsics.not_implemented()


@hipy.compiled_function
def sorted(input):
    l = list(input)
    l.sort()
    return l


@hipy.compiled_function
def abs(x):
    if intrinsics.isa(x, int):
        return x if x >= 0 else -x
    elif intrinsics.isa(x, float):
        return x if x >= 0.0 else -x
    else:
        intrinsics.not_implemented()