import hipy
import hipy.ir as ir
from hipy import binding
from hipy.value import ValueHolder, HLCClassValue, CValue, VoidValue, TypeValue, RawValue, MaterializedFunction, \
    LambdaValue, Value, Type, MaterializedConstantValue, HLCFunctionValue


def constify(arg):
    from hipy.lib.builtins import _concrete_list,_concrete_dict
    match arg:
        case ValueHolder(value=arg):
            match arg:
                case CValue(cval=arg):
                    return arg
                case HLCClassValue(cls=cls):
                    return cls.__hipy_create_type__()
                case TypeValue(type=type):
                    return type
                case _concrete_list(items=args):
                    return [constify(a) for a in args]
                case hipy.lib.builtins.tuple(_elts=args):
                    return tuple([constify(a) for a in args])
                case _concrete_dict(c_dict=cdict):
                    return {k: constify(v) for k, v in cdict.items()}
                case _:
                    return None
        case _:
            raise RuntimeError(f"Invalid argument for create_type: {arg}")
@hipy.raw
def call_builtin(fn_name, return_type, args, side_effects=None,attributes=None, _context=None):
    match side_effects:
        case None:
            side_effects = True
        case ValueHolder(value=CValue(cval=side_effects)):
            pass
        case _:
            raise RuntimeError(f"Invalid argument for side_effects: {side_effects}")
    from hipy.lib.builtins import _concrete_list
    match fn_name:
        case ValueHolder(value=CValue(cval=fn_name)):
            pass
        case _:
            raise RuntimeError(f"Invalid argument for function name {fn_name}")
    match args:
        case ValueHolder(value=_concrete_list(items=args)):
            pass
        case _:
            raise RuntimeError(f"Invalid argument for function args {args}")
    match return_type:
        case ValueHolder(value=return_type_val):
            match return_type_val:
                case VoidValue():
                    return_type = None
                case HLCClassValue(cls=cls):
                    return_type = cls.__hipy_create_type__()
                case TypeValue(type=type):
                    return_type = type
                case _:
                    raise RuntimeError(f"Invalid argument for function return type {return_type}")
    if attributes is not None:
        attributes=constify(attributes)
    return ValueHolder(_context.call_builtin(fn_name, return_type, args, side_effects=side_effects,attributes=attributes), _context)


@hipy.raw
def create_type(cls, *args, _context):


    type_args = []
    for arg in args:
        type_args.append(constify(arg))
    return _context.wrap(TypeValue(cls.value.cls.__hipy_create_type__(*type_args)))


@hipy.raw
def to_python(value, _context):
    return _context.to_python(value)

@hipy.raw
def try_narrow(value,type, _context):
    if isa(value,type,_context).value.cval:
        return value
    else:
        try:
             value.value.__narrow_type__(value, type.value.type)
        except (AttributeError, NotImplementedError):
            return value


@hipy.raw
def isa(value, cls, _context):
    match value, cls:
        case ValueHolder(value=v), ValueHolder(value=HLCClassValue(cls=cls)):
            return _context.constant(isinstance(v, cls), 0)
        case ValueHolder(value=v), ValueHolder(value=TypeValue(type=type)):
            #if v.__hipy_get_type__() != type:
            #    print("not the same:", v.__hipy_get_type__(), type)
            return _context.constant(v.__hipy_get_type__() == type, 0)
        case _:
            raise RuntimeError(f"Invalid arguments for isa: {value}, {cls}")


@hipy.raw
def not_implemented(_context):
    raise NotImplementedError()


@hipy.raw
def only_implemented_if(*args, _context):
    for i, arg in enumerate(args):
        match arg.value:
            case CValue(cval=arg):
                if not arg:
                    raise NotImplementedError(f"only_implemented_if failed because of argument {i}")
            case _:
                raise RuntimeError(f"Invalid argument for only_implemented_if: {arg}")


@hipy.raw
def import_pymodule(name, _context):
    match name:
        case ValueHolder(value=CValue(cval=name)):
            return _context.pyobj(ir.PyImport(_context.block, name).result)
    raise RuntimeError(f"Invalid argument for import_pymodule: {name}")


@hipy.raw
def typeof(value, _context):
    t = value.value.__hipy_get_type__()
    return ValueHolder(TypeValue(t), _context)


@hipy.raw
def isoftype(value, type, _context):
    match value, type:
        case ValueHolder(value=v), ValueHolder(value=TypeValue(type=type)):
            return _context.constant(v.__hipy_get_type__() == type, 0)
        case _:
            raise RuntimeError(f"Invalid arguments for isoftype: {value}, {type}");


@hipy.raw
def track_nested(nested, container, _context):
    return _context.track_nested(nested, container)


@hipy.raw
def reinterpret(value, return_type, _context):
    value = value.as_abstract(_context)
    match return_type:
        case ValueHolder(value=return_type_val):
            match return_type_val:
                case VoidValue():
                    return_type = None
                case HLCClassValue(cls=cls):
                    return_type = cls.__hipy_create_type__()
                case TypeValue(type=type):
                    return_type = type
                case _:
                    raise RuntimeError(f"Invalid argument for function return type {return_type}")
    match value:
        case ValueHolder(value=v):
            return _context.wrap(return_type.construct(v.__value__, _context))
        case _:
            raise RuntimeError(f"Invalid arguments for reinterpret: {value}");


bound_ctr = 0

@hipy.raw
def bind(fn, arg_types, _context):
    global bound_ctr

    from hipy.lib.builtins import _concrete_list
    import hipy.lib as lib
    fn_object = fn
    match fn, arg_types:
        case ValueHolder(value=fn), ValueHolder(value=_concrete_list(items=arg_types_)):
            arg_types = []
            for a in arg_types_:
                match a.value:
                    case TypeValue(type=t):
                        arg_types.append(t)
                    case HLCClassValue(cls=cls):
                        arg_types.append(cls.__hipy_create_type__())
                    case _:
                        raise RuntimeError(f"Invalid argument for bind: {fn}, {arg_types}")
            match fn:
                case LambdaValue(bind_staged=bind_staged_fn):


                    def bind_fn(closure_dict, fn):
                        global bound_ctr
                        closure=binding.create_closure(closure_dict,_context)
                        closure_type = closure.value.__hipy_get_type__()
                        explict_closure_arg = not isinstance(closure_type.ir_type(), ir.VoidType)
                        func = ir.Function(_context.module, f"bound_fn{bound_ctr}",
                                           [t.ir_type() for t in arg_types] + ([closure_type.ir_type()] if explict_closure_arg else []), ir.void)
                        bound_ctr += 1

                        with _context.use_block(func.body):
                            args = [
                                _context.wrap(t.construct(arg, _context)) for
                                t, arg in zip(arg_types, func.args)]
                            closure_param=func.args[-1] if explict_closure_arg else ir.Constant(_context.block, 0,ir.void).result
                            closure_param = _context.wrap(closure_type.construct(closure_param, _context))
                            res = fn(*args, _context=_context, __closure__=closure_param)
                            ir_res = res.get_ir_value(_context)
                            res_type = res.value.__hipy_get_type__()
                            ir.Return(_context.block, [ir_res])
                        func.res_type = ir_res.type
                        return _context.wrap(MaterializedFunction(
                            ir.FunctionRef(_context.block, func, closure.get_ir_value(_context) if explict_closure_arg else None).result ,
                            _context.wrap(TypeValue(res_type))))

                    return bind_staged_fn(bind_fn)
                case HLCFunctionValue(fn=hlc_fn):
                    func = ir.Function(_context.module, f"bound_fn{bound_ctr}",
                                       [t.ir_type() for t in arg_types], ir.void)
                    bound_ctr += 1

                    with _context.use_block(func.body):
                        args = [
                            _context.wrap(t.construct(arg, _context)) for
                            t, arg in zip(arg_types, func.args)]
                        res = hlc_fn.get_compiled_fn()(*args, _context=_context)
                        ir_res = res.get_ir_value(_context)
                        res_type = res.value.__hipy_get_type__()
                        ir.Return(_context.block, [ir_res])
                    func.res_type = ir_res.type
                    return _context.wrap(MaterializedFunction(
                        ir.FunctionRef(_context.block, func).result,
                        _context.wrap(TypeValue(res_type))))
                case lib.builtins.object():
                    func = ir.Function(_context.module, f"bound_fn{bound_ctr}",
                                       [t.ir_type() for t in arg_types]+[ir.pyobj], ir.void)
                    bound_ctr += 1
                    with _context.use_block(func.body):
                        args = [
                            _context.wrap(t.construct(arg, _context)) for
                            t, arg in zip(arg_types, func.args)]
                        callable = _context.wrap(fn.__hipy_get_type__().construct(func.args[-1],_context))
                        res = _context.perform_call(callable, args)
                        ir_res = res.get_ir_value(_context)
                        res_type = res.value.__hipy_get_type__()
                        ir.Return(_context.block, [ir_res])
                    func.res_type = ir_res.type
                    return _context.wrap(MaterializedFunction(
                        ir.FunctionRef(_context.block, func, fn_object.get_ir_value(_context)).result,
                        _context.wrap(TypeValue(res_type))))

                case _:
                    raise RuntimeError(f"Invalid argument for bind: {fn}, {arg_types}")

        case _:
            raise RuntimeError(f"Invalid arguments for bind: {fn}, {arg_types}")

@hipy.raw
def error(msg,_context):
    match msg:
        case ValueHolder(value=CValue(cval=msg)):
            raise RuntimeError(msg)
        case _:
            raise RuntimeError(f"error(): Invalid argument for error: {msg}")

@hipy.compiled_function
def try_or_default(fn, default):
    bound_fn=bind(fn, [])
    if bound_fn.res_type != typeof(default):
        error("try_or_default: return type of fn does not match default")
    return call_builtin("try_or_default", bound_fn.res_type, [bound_fn,default])

@hipy.raw
def create_record(names, values, _context):
    from hipy.lib.builtins import _concrete_list
    raw_names=[]
    match names:
        case ValueHolder(value=_concrete_list(items=args)):
            for a in args:
                match a.value:
                    case CValue(cval=a):
                        raw_names.append(a)
                    case _:
                        raise RuntimeError(f"Invalid argument for create_record: {names}")
            pass
        case _:
            raise RuntimeError(f"Invalid argument for create_record: {names}")
    match values:
        case ValueHolder(value=_concrete_list(items=args)):
            assert(len(args)==len(raw_names))
        case _:
            raise RuntimeError(f"Invalid argument for create_record: {values}")
    return _context.wrap(RawValue(_context.make_record(raw_names, args)))

@hipy.raw
def as_constant(value, _context):
    return _context.wrap(MaterializedConstantValue(value.value))


@hipy.raw
def undef(type,_context):
    match type:
        case ValueHolder(value=TypeValue(type=type)):
            return _context.wrap(type.construct(ir.Undef(_context.block,type.ir_type()).result,_context))
        case _:
            raise RuntimeError(f"Invalid argument for undef: {type}")
@hipy.raw
def call_indirect(fn, args,res_type,_context):
    from hipy.lib.builtins import _concrete_list

    fn=fn.value.__value__
    match res_type:
        case ValueHolder(value=TypeValue(type=type)):
            res_type=type
        case _:
            raise RuntimeError(f"Invalid argument for call_indirect: {res_type}")
    match args:
        case ValueHolder(value=_concrete_list(items=args)):
            pass
        case _:
            raise RuntimeError(f"Invalid argument for call_indirect: {args}")
    args=[a.get_ir_value(_context) for a in args]
    return _context.wrap(res_type.construct(ir.CallIndirect(_context.block, fn, args, res_type.ir_type()).result,_context))


@hipy.raw
def as_abstract(value, _context):
    return value.as_abstract(_context)
