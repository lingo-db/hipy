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
    def __init__(self, value, known_object=None, abstract_path=None):
        super().__init__(value)
        self._known_object = known_object
        self._abstract_path = abstract_path

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
            if power < 5:
                res = self
                for i in range(1, power):
                    res = res * self
                return res
        return intrinsics.call_builtin("python.operator.pow", object, [self, power])

    @hipy.raw
    def __call__(self, *args, _context, **kwargs):

        def get_type_for_ast(definition):
            match definition:
                case ast.FunctionDef(returns=ast.Name(id=name)):
                    match name:
                        case 'str' | 'int' | 'float':
                            return name
                        case 'LiteralString':
                            return 'str'
                    return name

        def get_converter_for_type(t):
            match t:
                case "int":
                    return int.__hipy_create_type__(), "scalar.int.from_python"
                case "float":
                    return float.__hipy_create_type__(), "scalar.float.from_python"
                case "bool":
                    return bool.__hipy_create_type__(), "scalar.bool.from_python"
                case "str":
                    return str.__hipy_create_type__(), "scalar.string.from_python"
            return None, None

        def infer_return_type_typeshed(name):
            import typeshed_client
            resolver = typeshed_client.Resolver()
            splitted = name.rsplit(".", 1)
            classname = splitted[0]
            membername = splitted[1]
            resolved = resolver.get_fully_qualified_name(classname)
            if not resolved:
                return None
            child = resolved.child_nodes[membername]
            if not child:
                return None
            child_ast = child.ast
            if not child_ast:
                return None
            match child_ast:
                case typeshed_client.OverloadedName(definitions=defs):
                    shared_type = None
                    for definition in defs:
                        t = get_type_for_ast(definition)
                        if shared_type is None:
                            shared_type = t
                        else:
                            if shared_type != t:
                                return None
                    return shared_type
                case ast.FunctionDef():
                    return get_type_for_ast(child_ast)

        def try_infer_return_type():
            if self.value._known_object is not None:
                fn = self.value._known_object
                try:
                    fn_source = inspect.getsource(fn)
                    fn_ast = ast.parse(fn_source)
                    match fn_ast:
                        case ast.Module(body=[definition]):
                            return get_converter_for_type(get_type_for_ast(definition))
                    return None, None
                except:
                    return None, None
            elif self.value._abstract_path is not None:
                return get_converter_for_type(infer_return_type_typeshed(self.value._abstract_path))
            else:
                return None, None

        to_python = lambda v: _context.to_python(v)
        py_res = ValueHolder(object(
            ir.PythonCall(_context.block, self.get_ir_value(_context),
                          [to_python(a).get_ir_value(_context) for a in args],
                          [(n, to_python(a).get_ir_value(_context)) for n, a in kwargs.items()], ).result),
            _context)
        ret_type, conversion = try_infer_return_type()
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
                return ValueHolder(object(ir.PyGetAttr(_context.block, attr, self.get_ir_value(_context)).result,
                                          abstract_path=f"{self.value._abstract_path}.{attr}" if self.value._abstract_path else None),
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
        elif intrinsics.isa(other, bool):
            return self._int_op(op, int(other), reverse)
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
    def __lshift__(self, other):
        return self._int_op("lshift", other)

    @hipy.compiled_function
    def __neg__(self):
        return 0 - self

    @hipy.compiled_function
    def __invert__(self):
        return self._int_op("xor", -1)

    @hipy.compiled_function
    def __and__(self, other):
        return self._int_op("and", other)

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

    @hipy.compiled_function
    def __iand__(self, other):
        return self & other

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
        elif intrinsics.isa(val, str):
            return intrinsics.call_builtin("scalar.float.from_string", float, [val])
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

    @hipy.compiled_function
    def __bool__(self):
        return self != 0.0

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
        return intrinsics.annotate_object_abstract_path(
            intrinsics.call_builtin("scalar.string.to_python", object, [self]), "builtins.str")

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
            res = res + v  # todo: if python: "cast to str"
        return res

    @hipy.compiled_function
    def __hipy__repr__(self):
        return "'" + self + "'"

    @hipy.compiled_function
    def __format__(self, fmt):
        return self

    @hipy.compiled_function
    def __len__(self):
        return intrinsics.call_builtin("scalar.string.length", int, [self])

    @hipy.compiled_function
    def split(self, pattern=None, maxsplit=-1):
        if pattern is None:
            res = []
            curr = ""
            for c in self:
                o = ord (c)
                if (o>= 9 and o <=13) or (o >= 28 or o <=32) or o == 160 or o == 5760 or (o >= 8192 and o<= 8202) or o == 8232 or o == 8233 or o == 8239 or o == 8287 or o == 12288:
                    if curr:
                        res.append(curr)
                        curr = ""
                else:
                    curr = curr + c
            return res
        else:
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

    @hipy.compiled_function
    def count(self, sub, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self)
        c = 0
        while start < end:
            pos = self.find(sub, start, end)
            if pos == -1:
                start = end
            else:
                c += 1
                start = pos + len(sub)
        return c


@hipy.classdef
class _const_str(CValue, str):
    __HIPY_MATERIALIZED__ = False

    def __init__(self, cval):
        str.__init__(self, None)
        CValue.__init__(self, cval)

    def __abstract__(self, _context):
        return str(ir.Constant(_context.block, self.cval, ir.string).result)

    @staticmethod
    def translate_python_spec_to_cpp(py: str) -> str:
        """
        Translate a single Python 'format_spec' into a C++20 std::format specifier.
        Very incomplete; handles only the easy direct mappings.
        """
        import re

        # Empty spec → empty C++ spec
        if py == "":
            return ""

        # Regex for the common subset:
        #
        # fill? align? sign? alt? zero? width? precision? type?
        #
        # We keep this deliberately conservative.
        spec_re = re.compile(
            r"""
            (?:(?P<fill>.)?(?P<align>[<>=^]))?
            (?P<sign>[+\- ])?
            (?P<alt>\#)?
            (?P<zero>0)?
            (?P<width>\d+)?
            (?:\.(?P<prec>\d+))?
            (?P<type>[bcdeEfFgGnosxX%])?
            """,
            re.VERBOSE
        )

        m = spec_re.fullmatch(py)
        if not m:
            raise NotImplementedError(f"Cannot parse Python format spec: {py}")

        gd = m.groupdict()

        fill = gd["fill"]
        align = gd["align"]
        sign = gd["sign"]
        alt = gd["alt"]
        zero = gd["zero"]
        width = gd["width"]
        prec = gd["prec"]
        typ = gd["type"]

        # Unsupported types / differences:
        if typ == "n":  # locale-aware number formatting has no C++ equivalent
            raise NotImplementedError("Python 'n' format has no C++ equivalent.")

        if align == "=":
            raise NotImplementedError("Python sign-aware alignment '=' not supported in C++20.")

        # Build a minimal C++ format string:
        out = "{:"

        # fill+align
        if align:
            if fill:
                out += fill + align
            else:
                out += align

        # sign
        if sign:
            out += sign

        # alt (#)
        if alt:
            out += "#"

        # zero
        if zero and width:
            # In both Python & C++ zero works the same
            out += "0"

        # width
        if width:
            out += width

        # precision
        if prec:
            out += "." + prec

        # type
        if typ:
            out += typ

        out += "}"

        return out

    @hipy.raw
    def __get_format_parts(self, _context):
        import re

        class FormatParseError(Exception):
            pass

        def parse_and_translate_format(fmt: str):
            """
            Parse a Python format string `fmt` into:
              - literals: list of n+1 literal strings
              - specs:    list of n raw specifiers (without braces)
              - cpp_specs: list of n translated C++20 specifiers

            Restrictions:
              * Field names are NOT allowed; if present → NotImplementedError.
              * Only a limited subset of Python specs is translated:
                    fill/align, width, precision, integer bases, float formats.
              * Anything outside this subset → NotImplementedError.
            """

            # Step 1: separate literals and specifiers using regex
            token_re = re.compile(r"{([^}]*)}")
            pos = 0
            literals = []
            specs = []

            for m in token_re.finditer(fmt):
                start, end = m.span()
                # literal before spec
                literals.append(fmt[pos:start])
                spec = m.group(1)
                specs.append(spec)
                pos = end

            # tail literal
            literals.append(fmt[pos:])

            # Step 2: validate + translate specifiers
            cpp_specs = []

            for spec in specs:
                # Python format field syntax may be:
                #   field_name[!conversion][:format_spec]
                # We do *not* handle field names → must be empty or begin with ':'
                if "!" in spec:
                    raise NotImplementedError("Conversion flags (!r, !s, ...) not supported.")

                # Split field name from format spec
                if ":" in spec:
                    field_name, fmt_spec = spec.split(":", 1)
                else:
                    field_name, fmt_spec = spec, ""

                if field_name.strip():
                    raise NotImplementedError("Field names are not supported.")

                # fmt_spec now contains Python's format-spec language.
                cpp_specs.append(_const_str.translate_python_spec_to_cpp(fmt_spec))

            return literals, specs, cpp_specs

        const_str = self.value.cval

        literals, specs, cpp_specs = parse_and_translate_format(const_str)
        return _context.create_tuple([_context.create_list([_context.constant(s) for s in literals]),
                                      _context.create_list([_context.constant(s) for s in cpp_specs])])

    @hipy.raw
    def __get_percentage_format_parts(self, _context):
        from typing import List, Tuple

        def decompose_percent_format(fmt: str) -> Tuple[List[str], List[str]]:
            """
            Decompose an old-style '%' format string.

            Returns:
                literals, specs

                - literals: list of n+1 literal chunks (with '%%' unescaped to '%')
                - specs:    list of n normalised specs, each in Python .format
                            mini-language form so that `translate_python_spec_to_cpp`
                            can be reused.

            Raises:
                NotImplementedError:
                    - mapping keys:           '%(name)d'
                    - dynamic width/precision: '%*d', '%. *f'
                    - integer precision:      '%.3d', '%.4x', etc.
                    - length modifiers:       '%ld', '%hd', '%Lf'
                    - unsupported types:      '%r', '%', or anything not in
                                              'diouxXeEfFgGcrs%'
                ValueError:
                    - for incomplete trailing '%' specs.
            """
            literals: List[str] = []
            specs: List[str] = []

            i = 0
            n = len(fmt)
            last_literal_start = 0

            while i < n:
                if fmt[i] != '%':
                    i += 1
                    continue

                # Handle escaped %%
                if i + 1 < n and fmt[i + 1] == '%':
                    # It's part of the literal; we just skip over it and
                    # unescape later via .replace('%%', '%').
                    i += 2
                    continue

                # We hit a real '%' specifier.
                # Flush literal before it (unescaping %% -> %)
                literals.append(fmt[last_literal_start:i].replace('%%', '%'))
                i += 1  # skip '%'

                # Mapping keys like %(name)d are not supported
                if i < n and fmt[i] == '(':
                    raise NotImplementedError("Mapping keys like %(name)s are not supported")

                # ----- Parse flags -----
                flags_chars = '#0- +'
                flags = ''
                while i < n and fmt[i] in flags_chars:
                    flags += fmt[i]
                    i += 1

                # ----- Parse width -----
                # We do NOT support '*' (dynamic width); be conservative.
                width = ''
                if i < n and fmt[i] == '*':
                    raise NotImplementedError("Dynamic width ('*') is not supported")
                while i < n and fmt[i].isdigit():
                    width += fmt[i]
                    i += 1

                # ----- Parse precision -----
                precision = ''
                if i < n and fmt[i] == '.':
                    i += 1
                    if i < n and fmt[i] == '*':
                        raise NotImplementedError("Dynamic precision ('*') is not supported")
                    while i < n and fmt[i].isdigit():
                        precision += fmt[i]
                        i += 1

                # ----- Parse length modifiers -----
                # Be conservative: any occurrence of C length modifiers is rejected.
                length_mods = 'hlL'
                if i < n and fmt[i] in length_mods:
                    raise NotImplementedError("C length modifiers (h, l, L) are not supported")

                # ----- Type character -----
                if i >= n:
                    raise ValueError("Incomplete '%' format specifier at end of string")

                type_char = fmt[i]
                i += 1

                # Check allowed types in old-style %
                allowed_types = set("diouxXeEfFgGcrs%")
                if type_char not in allowed_types:
                    raise NotImplementedError(f"Unsupported '%' format type {type_char!r}")

                # Types we explicitly do NOT support translating
                if type_char in ('r', '%'):
                    # %r uses repr() (no direct format-equivalent),
                    # % type is special (prints a literal % with formatting)
                    raise NotImplementedError(f"Unsupported '%' format type {type_char!r}")

                # Normalise some legacy integer types to 'd'
                if type_char in ('i', 'u'):
                    type_char = 'd'

                # If this is an integer-like type and precision is specified,
                # we cannot faithfully represent that in the .format mini-language
                # ('.3d' is invalid there).
                int_types = set('doxX')
                if type_char in int_types and precision:
                    raise NotImplementedError(
                        f"Integer precision like '%.{precision}{type_char}' "
                        "cannot be mapped safely to .format()"
                    )

                # ----- Convert flags to .format-style mini-language -----
                # We construct: [fill][align][sign][#][0][width][.precision][type]
                align = ''
                sign = ''
                alt = False
                zero = False

                for f in flags:
                    if f == '-':
                        align = '<'  # left-align
                    elif f == '+':
                        sign = '+'
                    elif f == ' ' and not sign:
                        sign = ' '  # space sign, only if '+' not present
                    elif f == '#':
                        alt = True
                    elif f == '0':
                        zero = True

                # Left alignment overrides zero-padding
                if align == '<':
                    zero = False

                spec_parts: List[str] = []

                # [fill][align] – we only use default blank fill, so just put align
                if align:
                    spec_parts.append(align)

                # [sign]
                if sign:
                    spec_parts.append(sign)

                # [#]
                if alt:
                    spec_parts.append('#')

                # [0]
                if zero:
                    spec_parts.append('0')

                # [width]
                if width:
                    spec_parts.append(width)

                # [.precision]
                # For float and string types, precision is supported in .format.
                if precision:
                    float_types = set('eEfFgG')
                    if type_char in float_types or type_char == 's':
                        spec_parts.append('.' + precision)
                    else:
                        # Should be unreachable because we blocked int+precision above,
                        # but keep this as a safety net.
                        raise NotImplementedError(
                            f"Precision not supported for type {type_char!r} in .format()"
                        )

                # [type]
                spec_parts.append(type_char)

                specs.append(''.join(spec_parts))

                # Next literal starts after this spec
                last_literal_start = i

            # Trailing literal (after last spec), unescaping %%
            literals.append(fmt[last_literal_start:].replace('%%', '%'))

            return literals, specs

        const_str = self.value.cval
        literals, specs = decompose_percent_format(const_str)
        cpp_specs = [_const_str.translate_python_spec_to_cpp(spec) for spec in specs]
        return _context.create_tuple([_context.create_list([_context.constant(s) for s in literals]),
                                      _context.create_list([_context.constant(s) for s in cpp_specs])])

    @hipy.compiled_function
    def format(self, *args):
        literals, cpp_specs = self._const_str__get_format_parts()
        len = cpp_specs.__len__()
        res = literals[0]
        for i in range(len):
            res += intrinsics.call_builtin("scalar.string.format_single", str, [cpp_specs[i], args[i]])
            res += literals[i + 1]
        return res

    @hipy.compiled_function
    def __mod__(self, *args):
        literals, cpp_specs = self._const_str__get_percentage_format_parts()
        len = cpp_specs.__len__()
        res = literals[0]
        for i in range(len):
            res += intrinsics.call_builtin("scalar.string.format_single", str, [cpp_specs[i], args[i]])
            res += literals[i + 1]
        return res


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
        t = args[0]
        if isinstance(t, builtins.list):
            t = t[0]
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

    @hipy.compiled_function
    def index(self, item):
        ret = -1
        for i in range(len(self)):
            if self[i] == item:
                ret = i
                break
        if ret != -1:
            return ret
        intrinsics.call_builtin("error", None, ["ValueError: " + str(item) + " not in list"])
        return ret


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

    @hipy.raw
    def __mul__(self, multiplier, _context):
        if isinstance(multiplier.value, _const_int):
            res = self.value.items * multiplier.value.cval
            return _context.wrap(_concrete_list(res))
        else:
            return _context.perform_call(_context.get_attr(self.as_abstract(_context), "__mul__"),
                                         [multiplier])

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
        def __iter__(self, _context):
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
                return intrinsics.create_type(tuple, [self._dict_val._key_type, self._dict_val._value_type])

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
    return intrinsics.bind(lambda l, r: l == r, [key_type, key_type])


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
        dict_type = self.__hipy_get_type__()
        key_type = _context.wrap(TypeValue(self._key_type))
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
        return intrinsics.create_type(tuple, [int, self.iterable.__itertype__()])

    @hipy.compiled_function
    def __iterate__(self, loopfn, x, iter_vals):
        # todo: maybe we should use a different type for the mutable counter...
        counter = [0]

        def wrapperfn(x, iter_vals, i):
            new_res = intrinsics.call_indirect(loopfn, [x, iter_vals, (counter[0], i)], intrinsics.typeof(iter_vals))
            counter[0] += 1
            return new_res

        wrapped_fn = intrinsics.bind(wrapperfn,
                                     [intrinsics.typeof(x), intrinsics.typeof(iter_vals), self.iterable.__itertype__()])
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

    @hipy.compiled_function
    def __str__(self):
        return repr(self)

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


@hipy.raw
def _format_internal(value, formatspec, _context=None):
    with _context.no_fallback():
        return _context.perform_call(_context.get_attr(value, "__format__"), [formatspec])


@hipy.compiled_function
def format(value, formatspec=""):
    return _format_internal(value, formatspec)


@hipy.compiled_function
def round(value, ndigits=None):
    if intrinsics.isa(value, float):
        if ndigits is None:
            return intrinsics.call_builtin("scalar.float.round", int, [value, 0])
        else:
            return intrinsics.call_builtin("scalar.float.round", float, [value, ndigits])
    elif intrinsics.isa(value, int):
        if ndigits is None:
            return value
        else:
            intrinsics.not_implemented()
            return value
    else:
        intrinsics.not_implemented()


@hipy.classdef
class _MaybeNone(static_object["_isNone", "_val"]):
    def __init__(self, isNone, val):
        super().__init__(lambda args: _MaybeNone(*args), isNone, val)

    @staticmethod
    @hipy.raw
    def __create__(isNone, val, _context=None):
        return hipy.value.ValueHolder(_MaybeNone(isNone, val), _context)

    @hipy.compiled_function
    def __topython__(self):
        if self._isNone:
            return intrinsics.call_builtin("python.get_none", object, [])
        else:
            return intrinsics.to_python(self._val)

    @hipy.compiled_function
    def __hipy_getattr__(self, name):
        if self._isNone:
            intrinsics.call_builtin("error", None,
                                    ["AttributeError: 'NoneType' object has no attribute '" + name + "'"])
        return intrinsics.get_attr(self._val, name)

    @hipy.compiled_function
    def __hipy__repr__(self):
        if self._isNone:
            return 'None'
        else:
            return self._val.__repr__()

    @hipy.compiled_function
    def __str__(self):
        if self._isNone:
            return 'None'
        else:
            return self._val.__str__()

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, _MaybeNone):
            return self, other, lambda val: self.value.__hipy_get_type__().construct(val, context)
        else:
            raise NotImplementedError()

    @hipy.compiled_function
    def __bool__(self):
        if self._isNone:
            return False
        else:
            return bool(self._val)
