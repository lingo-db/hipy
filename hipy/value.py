import ast
import copy
import inspect
from abc import abstractmethod, ABC
import hipy.ir as ir
import hipy.decorators
from hipy.function import HLCFunction


class Type(ABC):

    @abstractmethod
    def ir_type(self):
        pass

    @abstractmethod
    def construct(self, value, context):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    def get_cls(self):
        return None


class AnyType(Type):
    def ir_type(self):
        return ir.pyobj

    def construct(self, value, context):
        from hipy.lib.builtins import object as obj
        return obj(value)

    def __eq__(self, other):
        return isinstance(other, AnyType)

    def __repr__(self):
        return f"AnyType()"


class SimpleType(Type):
    def __init__(self, cls, ir_type):
        self.cls = cls
        self._ir_type = ir_type

    def ir_type(self):
        return self._ir_type

    def construct(self, value, context):
        return self.cls(value)

    def __eq__(self, other):
        return isinstance(other, SimpleType) and self.cls == other.cls and self._ir_type == other._ir_type

    def __repr__(self):
        return f"SimpleType({self.cls}, {self._ir_type})"

    def get_cls(self):
        return self.cls




class Value(ABC):
    __HIPY_MUTABLE__ = True
    __HIPY_NESTED_OBJECTS__ = True
    __HIPY_MATERIALIZED__ = True

    def __init__(self, value: ir.SSAValue):
        if value is not None:
            assert isinstance(value, ir.SSAValue)
        self.__value__ = value

    def __ir_value__(self):
        return self.__value__

    def __abstract__(self, context):
        return self

    @abstractmethod
    def __topython__(self):
        pass

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        raise NotImplementedError()
        """

        :param other: other value
        :param self_fn: callable object that takes a function f as input where f is a function that takes a context and returns a value
        :param other_fn: callable object that takes a function f as input where f is a function that takes a context and returns a value
        :param context: context
        :return: IR-compatible values of self and other, and a function f that takes an ir value and produces a merged Value
        """

    @staticmethod
    @abstractmethod
    def __hipy_create_type__(*args) -> Type:
        """
        creates a type according to the type arguments
        will get called when
        """

    @abstractmethod
    def __hipy_get_type__(self) -> Type:
        """
        returns the type of this value
        """


@hipy.decorators.classdef
class RawValue(Value):
    def __init__(self, value):
        super().__init__(value)

    def __topython__(self):
        raise RuntimeError("Cannot convert raw value to python")

    def __merge__(self, other, self_fn, other_fn, context):
        raise RuntimeError("Cannot merge raw values")

    class RawValueType(Type):
        def __init__(self, ir_type):
            self._ir_type = ir_type

        def ir_type(self):
            return self._ir_type

        def construct(self, value, context):
            return RawValue(value)

        def __repr__(self):
            return f"RawValueType)"

        def __eq__(self, other):
            return isinstance(other, self.__class__) and self._ir_type == other._ir_type

    @staticmethod
    def __hipy_create_type__(t) -> Type:
        return RawValue.RawValueType(t)

    def __hipy_get_type__(self) -> Type:
        return RawValue.RawValueType(self.__value__.type)

@hipy.decorators.classdef
class MaterializedConstantValue(Value):
    def __init__(self, o_value,value=None):
        self._value = o_value
        super().__init__(value)
    @hipy.raw
    def __topython__(self, _context):
        raise NotImplementedError()

    class MaterializedConstantValueType(Type):
        def __init__(self, value):
            self.value = value
        def ir_type(self):
            return ir.void
        def construct(self, value, context):
            return self.value
        def __eq__(self, other):
            return isinstance(other, self.__class__) and self.value==other.value
    def __abstract__(self, context):
        return context.wrap(MaterializedConstantValue(self._value,value=ir.Constant(context.block, None, ir.void).result))

    @staticmethod
    def __hipy_create_type__(value, *args):
        return MaterializedConstantValue.MaterializedConstantValueType(value)

    def __hipy_get_type__(self):
        return MaterializedConstantValue.MaterializedConstantValueType(self._value)




@hipy.decorators.classdef
class ConstIterValue(Value):
    def __init__(self, over, values):
        self.over = over
        self.values = values
        super().__init__(None)

    def __abstract__(self, context):
        raise NotImplementedError()

    @hipy.raw
    def __topython__(self, _context):
        return _context.to_python(self.as_abstract(_context))

    def __hipy_create_type__(*args) -> Type:
        raise NotImplementedError()

    def __hipy_get_type__(self) -> Type:
        raise NotImplementedError()


class ValueHolder:
    def __init__(self, value: Value, context):
        self.context = context
        self.t_location = tuple(context.location)
        self.value = value
        if context.debug:
            assert self.t_location not in context.seen
            context.seen.add(self.t_location)
        if self.t_location in context.decisions:
            with context.handle_action(context.invalid_action_id):
                self.value = context.to_python(self).value
    class AbstractViaPython:
        pass
    def __setattr__(self, key, value):
        if key == "value":
            self.context.log_set_value(self)

        super().__setattr__(key, value)

    def as_abstract(self, context):
        abstract = self.value.__abstract__(context)
        match abstract:
            case ValueHolder(value=abstract):
                # todo: track abstract values
                pass
            case Value():
                pass
            case ValueHolder.AbstractViaPython():
                return context.to_python(self)
            case _:
                raise RuntimeError(f"Invalid abstract value {abstract}")
        if self.value.__HIPY_MUTABLE__:
            self.value = abstract
            assert (self.value.__ir_value__() is not None)
            return self
        else:
            return context.wrap(abstract)
    def get_ir_value(self, context=None):
        if self.value.__ir_value__() is not None:
            return self.value.__ir_value__()
        else:
            return self.as_abstract(self.context).get_ir_value(self.context)


@hipy.decorators.classdef
class VoidValue(Value):
    __HIPY_MUTABLE__ = False
    def __init__(self, value=None):
        super().__init__(value)

    @hipy.raw
    def __topython__(self, _context):
        import hipy.intrinsics as intrinsics
        from hipy.lib.builtins import object as pyobject
        return _context.wrap(_context.call_builtin("python.get_none", pyobject.__hipy_create_type__(), []))

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, VoidValue):
            # todo: optimization potential
            return self, other, lambda val: VoidValue(val)
        else:
            raise NotImplementedError()

    class VoidType(Type):
        def ir_type(self):
            return ir.VoidType()

        def construct(self, value,context):
            return VoidValue(value)

        def __eq__(self, other):
            if isinstance(other, VoidValue.VoidType):
                return True
            else:
                return False

        def __repr__(self):
            return f"VoidType()"

    def __abstract__(self, context):
        return context.wrap(VoidValue(ir.Constant(context.block, None, ir.void).result))

    def __hipy_create_type__(self, *args):
        return self.VoidType()

    def __hipy_get_type__(self):
        return self.VoidType()

    @hipy.compiled_function
    def __hipy__repr__(self):
        return "None"


@hipy.decorators.classdef
class CValue:
    def __init__(self, cval):
        self.cval = cval
    @staticmethod
    def _cvalue_binary_op(self, other, dunder, fn, _context):
        self_val = self.value
        match other:
            case ValueHolder(value=CValue(cval=other_cval)):
                return _context.constant(fn(self_val.cval,other_cval))
        return _context.perform_call(_context.get_attr(self, "_super_"+dunder), [other])
    @hipy.raw
    def __add__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__add__", lambda a, b: a + b, _context)

    @hipy.raw
    def __sub__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__sub__", lambda a, b: a - b, _context)

    @hipy.raw
    def __mul__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__mul__", lambda a, b: a * b, _context)

    @hipy.raw
    def __truediv__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__truediv__", lambda a, b: a / b, _context)

    @hipy.raw
    def __eq__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__eq__", lambda a, b: a == b, _context)

    @hipy.raw
    def __ne__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__ne__", lambda a, b: a != b, _context)
    @hipy.raw
    def __lt__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__lt__", lambda a, b: a < b, _context)
    @hipy.raw
    def __le__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__le__", lambda a, b: a <= b, _context)
    @hipy.raw
    def __gt__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__gt__", lambda a, b: a > b, _context)
    @hipy.raw
    def __ge__(self, other, _context):
        return CValue._cvalue_binary_op(self, other, "__ge__", lambda a, b: a >= b, _context)

    @hipy.raw
    def __contains__(self, item,_context):
        return CValue._cvalue_binary_op(self, item, "__contains__", lambda a, b: b in a,_context)

    @hipy.raw
    def __len__(self,_context):
        return _context.constant(len(self.value.cval))

    @hipy.raw
    def __neg__(self,_context):
        return _context.constant(-self.value.cval)



@hipy.decorators.classdef
class PythonModule(Value):
    def __init__(self, module, _raw=False):
        self.py_module = module
        if not _raw:
            from hipy import mocked_modules
            if module.__name__ in mocked_modules:
                module = mocked_modules[module.__name__]
        self._raw = _raw
        self.module = module

    @hipy.raw
    def __topython__(self, _context):
        return _context.pyobj(ir.PyImport(_context.block, self.value.py_module.__name__).result)

    def __getattr__(self, name):
        val = getattr(self.module, name)
        if inspect.ismodule(val):
            return PythonModule(val)
        else:
            match val:
                case HLCFunction():
                    return HLCFunctionValue(val)
                case _:
                    if inspect.isclass(val) and hasattr(val, "__hipy__"):
                        return HLCClassValue(val)
                    raise AttributeError()

    @hipy.raw
    def __hipy_getattr__(self, name, _context):
        if self.value._raw or _context.fallback():
            match name:
                case ValueHolder(value=CValue(cval=name_cval)):
                    return _context.get_attr(_context.to_python(self), name_cval)
        raise NotImplementedError()

    def __hipy_create_type__(self, *args):
        raise RuntimeError("Cannot create type of a module")

    def __hipy_get_type__(self):
        raise RuntimeError("Cannot get type of a module")


class RawModule:
    def __init__(self, module):
        self.module = module


def raw_module(module):
    return RawModule(module)


@hipy.decorators.classdef
class HLCClassValue(Value):
    def __init__(self, cls,value=None):
        self.cls = cls
        super().__init__(value)

    @hipy.raw
    def __topython__(self, _context):
        module = inspect.getmodule(self.value.cls)
        if hasattr(module, "__HIPY_MODULE__"):
            module_name = module.__HIPY_MODULE__
        else:
            module_name = module.__name__
        imported_module = _context.import_pymodule(module_name)
        class_name = self.value.cls.__name__
        # if module_name=="__main__":
        #    _context.module.py_functions[funcname] = remove_decorators(inspect.getsource(pyfunc))
        return _context.get_attr(imported_module, class_name)

    class HLCClassValueType(Type):
        def __init__(self,cls):
            self.cls=cls
        def ir_type(self):
            return ir.void
        def construct(self, value, context):
            return HLCClassValue(self.cls,value=value)
        def __eq__(self, other):
            return isinstance(other, self.__class__) and self.cls==other.cls
        def __repr__(self):
            return f"HLCClassValueType({self.cls})"
    def __hipy_create_type__(self, *args):
        return HLCClassValue.HLCClassValueType(self.cls)

    def __hipy_get_type__(self):
        return HLCClassValue.HLCClassValueType(self.cls)
    def __abstract__(self, context):
        return context.wrap(HLCClassValue(self.cls,value=ir.Constant(context.block, None, ir.void).result))

    def __getattr__(self, item):
        if item == "cls":
            return super().__getattr__(item)
        val = getattr(self.cls, item)
        return val
@hipy.decorators.classdef
class TypeValue(Value):
    """
    Just hold a raw python value that can not be used at all at runtime
    """

    def __init__(self, type: Type):
        self.type = type

    def __topython__(self, _context):
        raise NotImplementedError()

    class TypeValueType(Type):
        def __init__(self,type):
            self.type=type
        def ir_type(self):
            raise NotImplementedError()
        def construct(self, value, context):
            raise NotImplementedError()
        def __eq__(self, other):
            return isinstance(other, self.__class__) and self.type==other.type
        def __repr__(self):
            return f"TypeValueType({self.type})"

    @hipy.raw
    def __eq__(self, other,_context):
        if isinstance(other.value,TypeValue):
            return _context.constant(self.value.type==other.value.type)
        if isinstance(other.value, HLCClassValue):
            return _context.constant(self.value.type==other.value.cls.__hipy_create_type__())
        return False
    @hipy.compiled_function
    def __ne__(self, other):
        return not (self == other)


    def __hipy_create_type__(self, *args):
        return TypeValue.TypeValueType(*args)

    def __hipy_get_type__(self):
        return TypeValue.TypeValueType(self.type)

    @hipy.raw
    def __call__(self, *args,_context):
        type=self.value.type
        cls=type.get_cls()
        if cls is None or not hasattr(cls,"__create__"):
            raise NotImplementedError()
        return _context.perform_call(cls,args)


class MaterializedFunction(Value):
    def __init__(self, value, res_type):
        super().__init__(value)
        self.res_type=res_type

    def __topython__(self, _context):
        raise NotImplementedError()
    def __hipy_create_type__(*args):
        return SimpleType(MaterializedFunction, args[0])
    def __hipy_get_type__(self):
        return SimpleType(MaterializedFunction, self.__value__.type)


@hipy.decorators.classdef
class HLCGeneratorFunctionValue(Value):
    def __init__(self, fn, value=None):
        self.fn = fn
        super().__init__(value)

    @hipy.decorators.raw
    def __topython__(self, _context):
        def remove_decorators(fn):
            def is_hipy_decorator(d):
                try:
                    result = eval(compile(ast.unparse(d), filename='<string>', mode='eval'))
                    if inspect.getmodule(result).__name__.startswith("hipy"):
                        return True
                    else:
                        return False
                except Exception as e:
                    return False

            fn_ast = ast.parse(fn)
            decorator_list = fn_ast.body[0].decorator_list
            fn_ast.body[0].decorator_list = [d for d in decorator_list if not is_hipy_decorator(d)]
            return ast.unparse(fn_ast)

        hlc_function = self.value.fn
        pyfunc = hlc_function.pyfunc
        if pyfunc.__name__.startswith("_"):
            raise NotImplementedError("Cannot convert private functions to python")
        module = inspect.getmodule(pyfunc)
        if hasattr(module, "__HIPY_MODULE__"):
            module_name = module.__HIPY_MODULE__
        else:
            module_name = module.__name__
        imported_module = _context.import_pymodule(module_name)
        funcname = pyfunc.__name__
        if module_name == _context.base_module:
            _context.module.py_functions[funcname] = remove_decorators(inspect.getsource(pyfunc))
        return _context.get_attr(imported_module, funcname)

    class GeneratorFunctionValueType(Type):
        def __init__(self, fn):
            self.fn = fn

        def ir_type(self):
            return ir.void

        def construct(self, value, context):
            return HLCGeneratorFunctionValue(self.fn, value=value)

        def __eq__(self, other):
            return isinstance(other, self.__class__) and self.fn == other.fn

    def __abstract__(self, context):
        return context.wrap(HLCGeneratorFunctionValue(self.fn, value=ir.Constant(context.block, None, ir.void).result))

    @staticmethod
    def __hipy_create_type__(value, *args):
        return HLCGeneratorFunctionValue.GeneratorFunctionValueType(value)

    def __hipy_get_type__(self):
        return HLCGeneratorFunctionValue.GeneratorFunctionValueType(self.fn)

    @hipy.classdef
    class _iterator(Value):
        def __init__(self, func_args, value=None):
            super().__init__(value)
            self.func_args = func_args

        def __track__(self, iter_value, context):
            pass

        @hipy.compiled_function
        def __itertype__(self):
            return str #todo: how do we determine the type of the iterator?
        """
        hack:
            0. start "context" transaction
            1. create a "fake @raw function that collects the list of function types
            2. call the generator function with the real arguments and the fake callback function 
            3. merge the function types to one type
        """

        @hipy.compiled_function
        def __iterate__(self, loopfn, x, iter_vals):
            """
            hack:
                0. "nested function" that implicitely caputes the iter_values etc
                1. call the generator function with the real arguments and the "nested function"
            """

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
@hipy.decorators.classdef
class HLCFunctionValue(Value):
    def __init__(self, fn,value=None):
        self.fn = fn
        super().__init__(value)

    @hipy.decorators.raw
    def __topython__(self, _context):
        def remove_decorators(fn):
            def is_hipy_decorator(d):
                try:
                    result = eval(compile(ast.unparse(d), filename='<string>', mode='eval'))
                    if inspect.getmodule(result).__name__.startswith("hipy"):
                        return True
                    else:
                        return False
                except Exception as e:
                    return False

            fn_ast = ast.parse(fn)
            decorator_list = fn_ast.body[0].decorator_list
            fn_ast.body[0].decorator_list = [d for d in decorator_list if not is_hipy_decorator(d)]
            return ast.unparse(fn_ast)

        hlc_function = self.value.fn
        pyfunc = hlc_function.pyfunc
        if pyfunc.__name__.startswith("_"):
            raise NotImplementedError("Cannot convert private functions to python")
        module = inspect.getmodule(pyfunc)
        if hasattr(module, "__HIPY_MODULE__"):
            module_name = module.__HIPY_MODULE__
        else:
            module_name = module.__name__
        imported_module = _context.import_pymodule(module_name)
        funcname = pyfunc.__name__
        if module_name == _context.base_module:
            _context.module.py_functions[funcname] = remove_decorators(inspect.getsource(pyfunc))
        return _context.get_attr(imported_module, funcname)

    class FunctionValueType(Type):
        def __init__(self, fn):
            self.fn = fn
        def ir_type(self):
            return ir.void
        def construct(self, value, context):
            return HLCFunctionValue(self.fn,value=value)
        def __eq__(self, other):
            return isinstance(other, self.__class__) and self.fn==other.fn
    def __abstract__(self, context):
        return context.wrap(HLCFunctionValue(self.fn,value=ir.Constant(context.block, None, ir.void).result))

    @staticmethod
    def __hipy_create_type__(value, *args):
        return HLCFunctionValue.FunctionValueType(value)

    def __hipy_get_type__(self):
        return HLCFunctionValue.FunctionValueType(self.fn)


@hipy.decorators.classdef
class HLCMethodValue(Value):
    def __init__(self, fn, self_value):
        self.fn = fn
        self.self_value = self_value
        super().__init__(None)

    @hipy.raw
    def __topython__(self, _context):
        return _context.get_attr(_context.to_python(self.value.self_value), self.value.fn.func.pyfunc.__name__)

    def __hipy_create_type__(self, *args):
        raise NotImplementedError()

    def __hipy_get_type__(self):
        raise NotImplementedError()
class _NotRelevantType(Type):
    def ir_type(self):
        raise NotImplementedError()

    def construct(self, value, context):
        raise NotImplementedError()

    def __eq__(self, other):
        return False

    def __repr__(self):
        return "NotRelevantType"

@hipy.decorators.classdef
class LambdaValue(Value):
    def __init__(self, staged, bind_python, bind_staged):
        self.staged = staged
        self.bind_python = bind_python
        self.bind_staged = bind_staged
        super().__init__(None)

    @hipy.raw
    def __topython__(self, _context):
        def convert(closure_dict, fn, fn_name):
            _context.module.py_functions[fn_name] = fn
            _context.module.py_functions["_hipy_bind"] = """def _hipy_bind(fn, __closure__):
    return lambda *args, **kwargs: fn(*args, __closure__=__closure__, **kwargs)"""
            main_module = _context.pyobj(ir.PyImport(_context.block, "__main__").result)
            bind_fn = _context.get_attr(main_module, "_hipy_bind")
            fn = _context.get_attr(main_module, fn_name)
            closure_dict = _context.create_dict([_context.constant(k) for k in closure_dict.keys()],
                                                closure_dict.values())
            return _context.perform_call(bind_fn, [fn, closure_dict])

        res = self.value.bind_python(convert)
        return res

    def __hipy_create_type__(self, *args):
        return _NotRelevantType()

    def __hipy_get_type__(self):
        return _NotRelevantType()


class _static_object:
    def __getitem__(self, members):
        @hipy.classdef
        class static_object_value(hipy.value.Value):
            __HIPY_MUTABLE__ = False

            def __init__(self, constructor, *args, value=None):
                super().__init__(value)
                assert len(members) == len(args)
                assert all([isinstance(a, hipy.value.ValueHolder) for a in args])
                self._args = args
                self._constructor = constructor

            def __getattr__(self, item):
                if item == "_args":
                    return super().__getattr__(self, item)
                try:
                    return self._args[members.index(item)]
                except ValueError:
                    raise AttributeError()

            class static_object_value_type(hipy.value.Type):
                def __init__(self, types, constructor):
                    self.types = types
                    self.constructor = constructor

                def ir_type(self):
                    return ir.RecordType([(n, t.ir_type()) for n, t in zip(members, self.types)])

                def construct(self, value, context):
                    args = []
                    for n, t in zip(members, self.types):
                        args.append(context.wrap(
                            t.construct(ir.RecordGet(context.block, t.ir_type(), value, n).result, context)))
                    res = self.constructor(args)
                    assert isinstance(res, static_object_value)
                    res.__value__ = value
                    return res

                def __eq__(self, other):
                    return isinstance(other, self.__class__) and self.types == other.types

            @staticmethod
            def __hipy_create_type__(*args) -> Type:
                raise NotImplementedError()

            def __hipy_get_type__(self):
                return self.static_object_value_type([a.value.__hipy_get_type__() for a in self._args],
                                                     self._constructor)

            def __abstract__(self, context):
                if self.__value__ is not None:
                    return self
                else:
                    record_value = context.make_record(members, self._args)
                    return self.__hipy_get_type__().construct(record_value, context)

        return static_object_value


static_object = _static_object()
