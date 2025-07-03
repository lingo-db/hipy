import copy
import inspect
import sys

import hipy
from hipy import binding,global_const, mocked_modules
from hipy.value import Value, VoidValue, PythonModule, ValueHolder, HLCClassValue, CValue, HLCFunctionValue, \
    HLCMethodValue, LambdaValue, Type, TypeValue, RawValue, ConstIterValue, RawModule, HLCGeneratorFunctionValue
import hipy.ir as ir
import hipy.lib as lib
from hipy.function import HLCFunction, HLCMethod, GeneratorFunction


class ValueAlias:
    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2

    def __repr__(self):
        return f"{self.val1.t_location} = {self.val2.t_location}"


class NestedValueAlias:
    def __init__(self, nested, container):
        self.nested = nested
        self.container = container

    def __repr__(self):
        return f"{self.nested.t_location} in {self.container.t_location}"


class ConvertedToPython:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return f"to_python({self.val.t_location})"


class ValUsage:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return f"using {self.val.t_location}"


class DynAction:
    def __init__(self):
        self.counter = 0

    def id(self):
        self.counter += 1
        return f"dyn:{self.counter}"

class NewScope:
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        self.context.location.append(None)
        self.context.dyn_actions.append(DynAction())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.location.pop()
        self.context.dyn_actions.pop()
class Context:
    def unwrap(self, val):
        match val:
            case ValueHolder(value=val):
                return val
            case _:
                return val

    def wrap(self, val, _action_id=None):
        with self.handle_action(_action_id):
            match val:
                case ValueHolder():
                    assert False
                case _:
                    return ValueHolder(val, self)

    class EarlyReturn(Exception):
        def __init__(self, val):
            self.val = val

        def __repr__(self):
            return f"EarlyReturn({self.val})"

    def __init__(self, module: ir.Module, block: ir.Block, decisions=[], invalid_action_id=-1,debug=True, base_module="__main__"):
        # read-only state:
        self.invalid_action_id = invalid_action_id
        self.decisions = decisions
        self.debug = debug

        # transactional state: (have a look at transaction() method)
        self.module = module
        self.block = block
        self.events = []
        if self.debug:
            self.seen = set()

        # does not really matter:
        self.loop_fn_counter = 0
        self.transaction_cntr = 0
        self.try_cntr = 0

        # stacks (manged with withs)
        self.ignored_uses = []
        self.dyn_actions = []
        self.location = [None]
        self.no_fallbacks = 0
        self.active_transactions = []
        self.ignore_change=False
        self.base_module=base_module

    def log_set_value(self, val_holder):
        if len(self.active_transactions) > 0:
            self.active_transactions[-1].log_set_value(val_holder)
    def __setattr__(self, key, value):
        #if key == "block":
        #   print("setting block", value.id, file=sys.stderr)
        super().__setattr__(key, value)
    def fallback(self):
        return self.no_fallbacks == 0

    def no_fallback(self):
        class FallbackContext:
            def __init__(self, context):
                self.context = context

            def __enter__(self):
                self.context.no_fallbacks += 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.context.no_fallbacks -= 1

        return FallbackContext(self)

    def handle_action(self, action):
        if action is None:
            action = self.dyn_actions[-1].id()
        self.location[-1] = action



        return NewScope(self)

    def get_abstract_type(self, val):
        match val:
            case bool():
                return lib.builtins.bool
            case int():
                return lib.builtins.int
            case float():
                return lib.builtins.float
            case str():
                return lib.builtins.str
            case _:
                raise NotImplementedError()

    def pyobj(self, val):
        return ValueHolder(lib.builtins.object(val), self)

    def cval(self, v):
        match v:
            case bool():
                return lib.builtins._const_bool(v)
            case str():
                return lib.builtins._const_str(v)
            case int():
                return lib.builtins._const_int(v)
            case float():
                return lib.builtins._const_float(v)
            case bytes():
                return lib.builtins._const_bytes(v)
            case None:
                return VoidValue()
            case _:
                raise NotImplementedError()

    def constant(self, val, _action_id=None):
        with self.handle_action(_action_id):
            return self.wrap(self.cval(val))

    def perform_call(self, fn, args=[], kwargs={}, _action_id=None):
        # todo: handle fallback to python
        with self.handle_action(_action_id):
            original_fn = fn
            fn = self.unwrap(fn)
            assert all([isinstance(arg, ValueHolder) for arg in args])
            # args = [self.wrap(arg) for arg in args]
            try:
                match fn:
                    case type():
                        return self.perform_call(HLCFunctionValue(fn.__create__), args, kwargs)
                    case HLCClassValue(cls=cls):
                        return self.perform_call(self.wrap(HLCFunctionValue(cls.__create__)), args, kwargs)
                    case HLCFunctionValue(fn=fn):
                        #print("calling function", fn.pyfunc, args, kwargs, file=sys.stderr)
                        res = fn.get_compiled_fn()(*args, **kwargs, _context=self)
                        self.track_using(res)
                        return res
                    case HLCMethodValue(fn=HLCMethod(func=func), self_value=self_value):
                        assert isinstance(self_value, ValueHolder)
                        #print("calling method", func.pyfunc, args, kwargs, file=sys.stderr)
                        try:
                            res = func.get_compiled_fn()(self_value, *args, **kwargs, _context=self)
                            self.track_using(res)
                        except NotImplementedError as e:
                            if self.fallback():
                                res = self.perform_call(self.to_python(original_fn), args)
                                self.track_using(res)
                            else:
                                raise e
                        return res
                    case LambdaValue(staged=staged):
                        res = staged(*args, **kwargs, _context=self)
                        self.track_using(res)
                        return res
                    case Value():
                        if hasattr(fn, "__call__"):
                            return self.perform_call(self.get_attr(original_fn, "__call__"), args, kwargs)
                        else:
                            raise NotImplementedError()
                    case _:
                        raise NotImplementedError()
            except (AttributeError,TypeError, NotImplementedError) as e:
                if self.fallback() and not isinstance(fn, lib.builtins.object):
                    #print("falling back to python", fn, e, e.args, file=sys.stderr)
                    return self.perform_call(self.to_python(original_fn), args)
                else:
                    #print("failing to fallback to python", fn, e, e.args, file=sys.stderr)
                    raise e

    def early_return(self, val, _action_id):
        raise self.EarlyReturn(val)

    def call_builtin(self, fn, res, args, side_effects=True,attributes=None, _action_id=None):
        #print("calling builtin",self.block.id, fn, args, file=sys.stderr)
        with self.handle_action(_action_id):
            assert all([isinstance(arg, ValueHolder) for arg in args])
            arg_vals = [arg.get_ir_value(self) for arg in args]
            if res is None:
                return VoidValue(ir.CallBuiltin(self.block, fn, arg_vals, ir.void, side_effects).result)
            else:
                return res.construct(ir.CallBuiltin(self.block, fn, arg_vals, res.ir_type(), side_effects=side_effects,attributes=attributes).result, self)

    def perform_binop(self, left, right, left_method, right_method, _action_id=None):
        with self.handle_action(_action_id):
            # maybe we need to do this in a better way
            try:
                try:
                    with self.no_fallback():
                        return self.perform_call(self.get_attr(left, left_method), [right])
                except (NotImplementedError, TypeError, AttributeError):
                    if isinstance(right.value, lib.builtins.object):
                        raise NotImplementedError()
                    with self.no_fallback():
                        return self.perform_call(self.get_attr(right, right_method), [left])
                    # todo: handle fallback to other, more generic type (e.g. record -> dict)
            except (NotImplementedError, TypeError, AttributeError) as e:
                if self.fallback():
                    #print("falling back to python", left_method, file=sys.stderr)
                    try:
                        return self.perform_call(
                            self.get_attr(self.to_python(left), left_method),
                            [self.to_python(right)])
                    except Exception as e:
                        #print("failing to fallback to python", left_method, e, e.args, file=sys.stderr)
                        raise e
                else:
                    #print("fallback to python disabled", left_method, e, e.args, file=sys.stderr)
                    raise e

    def import_pymodule(self, name, _action_id=None):
        if name.startswith("hipy.") or name == "hipy":
            raise NotImplementedError()
        with self.handle_action(_action_id):
            if name == self.base_module:
                #todo: this should only be done once
                # for __main__ module, we need to import all modules from the global scope
                for k, v in sys.modules[ self.base_module].__dict__.items():
                    if inspect.ismodule(v):
                        module_name = v.__name__
                        # ignore dbpy module (should not be required at runtime)
                        if module_name.startswith("hipy.") or module_name == "hipy" or hasattr(v, "__HIPY_NO_IMPORT__"):
                            continue
                        # ignore __builtin__, , os, sys modules, otherwise there will be a clash
                        if k.startswith("__") or k == "os" or k == "sys":
                            continue
                        self.module.imports[k] = module_name
                main_module = self.pyobj(ir.PyImport(self.block, "__main__").result)
                return main_module
            else:
                return self.pyobj(ir.PyImport(self.block, name).result)

    def get_recursive_raw(self, func,module, _action_id=None):
        with self.handle_action(_action_id):
            return self.get_attr(self.get_by_name(sys.modules[module],"module"), func)
    def get_by_name(self, val, name, _action_id=None):
        with self.handle_action(_action_id):
            match val:
                case ValueHolder():
                    self.track_using(val)
                    return val
                case GeneratorFunction():
                    return self.wrap(HLCGeneratorFunctionValue(val))
                case HLCFunction():
                    return self.wrap(HLCFunctionValue(val))
                case Type():
                    return self.wrap(TypeValue(val))
                case RawModule(module=module):
                    return self.wrap(PythonModule(module, _raw=True))
                case global_const(value=val):
                    return self.constant(val)
                case str() | bytes():
                    return self.constant(val)
                case _:
                    if inspect.isclass(val) and hasattr(val, "__hipy__"):
                        return self.wrap(HLCClassValue(val))
                    if inspect.ismodule(val):
                        return self.wrap(PythonModule(val))
                    else:
                        val_module=val.__module__
                        if val_module in hipy.mocked_modules:
                            return self.get_attr(self.wrap(PythonModule(sys.modules[val_module])),name)
                        val_name= val.__name__ if hasattr(val, "__name__") else name

                        if val_module == self.base_module:
                            assert inspect.isfunction(val), f"only functions from {self.base_module} module are supported"
                            # get the source code of the python function and register it to the DBPyIR module
                            funcname = val.__name__
                            self.module.py_functions[funcname] = inspect.getsource(val)

                        imported_module=self.import_pymodule(val_module)

                        imported_obj = self.get_attr(imported_module, val_name)
                        return self.wrap(lib.builtins.object(known_object=val, value=imported_obj.get_ir_value(self)))
    def unpack(self, val, num, _action_id=None):
        with self.handle_action(_action_id):
            #todo: check if val is iterable
            res =[]
            for i in range(num):
                res.append(self.get_item(val, self.constant(i)))
            return res

    def const_unpack(self, val, _action_id=None):
        with self.handle_action(_action_id):
            with self.no_fallback():
                const_iter= self.perform_call(self.get_attr(val, "__constiter__"))
                match const_iter:
                    case ValueHolder(value=ConstIterValue(values=values)):
                        return values
        raise NotImplementedError()



    def set_item(self, val, item, value, _action_id=None):
        with self.handle_action(_action_id):
            return self.perform_call(self.get_attr(val, "__setitem__"), [item, value])

    def set_attr(self, val, attr, value, _action_id=None):
        with self.handle_action(_action_id):
            if hasattr(val.value, attr):
                setattr(val.value, attr, value)
                return
            self.perform_call(self.get_attr(val, "__hipy_setattr__"), [self.constant(attr), value])

    def get_item(self, val, item, _action_id=None):
        with self.handle_action(_action_id):
            try:
                return self.perform_call(self.get_attr(val, "__getitem__"), [item])
            except (NotImplementedError, TypeError, AttributeError) as e:
                if self.fallback():
                    return self.get_item(self.to_python(val), item)
                else:
                    raise e

    def get_attr(self, val, attr, _action_id=None):
        with self.handle_action(_action_id):
            val_holder = val
            val = self.unwrap(val)
            if hasattr(val, attr):
                attr_val = getattr(val, attr)
                match attr_val:
                    case ValueHolder():
                        self.track_using(attr_val)
                        return attr_val
                    case HLCFunction():
                        return self.wrap(HLCFunctionValue(attr_val))
                    case HLCMethod():
                        assert isinstance(val_holder, ValueHolder)
                        return self.wrap(HLCMethodValue(attr_val, val_holder))
                    case Value():
                        return self.wrap(attr_val)
                    case Type():
                        return self.wrap(TypeValue(attr_val))
            if hasattr(val, "__hipy_getattr__") and val.__hipy_getattr__ is not None:
                try:
                    return self.perform_call(self.get_attr(val_holder, "__hipy_getattr__"),
                                             [self.constant(attr)])
                except (NotImplementedError, TypeError, AttributeError) as e:
                    # todo: handle fallback to other, more generic type (e.g. record -> dict)
                    if self.fallback():
                        return self.get_attr(self.to_python(val_holder), attr)
                    else:
                        raise e

            else:
                # todo: handle fallback to other, more generic type (e.g. record -> dict)
                if self.fallback():
                    return self.get_attr(self.to_python(val_holder), attr)
                else:
                    raise AttributeError("no attribute", attr, "on", val)

    def perform_unaryop(self, fn, arg):
        try:
            return fn(arg)
        # todo: handle fallback to other, more generic type (e.g. record -> dict)
        except (NotImplementedError, TypeError):
            return fn(self.to_python(arg))

    def is_(self,left, right,_action_id=None):
        with self.handle_action(_action_id):
            match left.value, right.value:
                case (VoidValue(), VoidValue()):
                    return self.constant(True)
                case (VoidValue(), lib.builtins.object()):
                    raise NotImplementedError()
                case (lib.builtins.object(), VoidValue()):
                    raise NotImplementedError()
                case (_, VoidValue()):
                    return self.constant(False)
                case (VoidValue(), _):
                    return self.constant(False)
                case (lib.builtins.object(), lib.builtins.object()):
                    raise NotImplementedError()
        raise NotImplementedError()

    def in_(self,left, right, _action_id=None):
        with self.handle_action(_action_id):
            try:
                return self.perform_call(self.get_attr(right, "__contains__"), [left])
            except (NotImplementedError, TypeError, AttributeError) as e:
                if self.fallback():
                    return self.call_builtin("python.operator.contains", lib.builtins.bool, [right, left])
                else:
                    raise e

    def neg_(self,arg, _action_id=None):
        with self.handle_action(_action_id):
            try:
                return self.perform_call(self.get_attr(arg, "__neg__"), [])
            except (NotImplementedError, TypeError, AttributeError) as e:
                if self.fallback():
                    return self.call_builtin("python.operator.neg", lib.builtins.object, [arg])
                else:
                    raise e

    def invert_(self,arg, _action_id=None):
        with self.handle_action(_action_id):
            try:
                return self.perform_call(self.get_attr(arg, "__invert__"), [])
            except (NotImplementedError, TypeError, AttributeError) as e:
                if self.fallback():
                    return self.call_builtin("python.operator.inv", lib.builtins.object, [arg])
                else:
                    raise e


    def _to_bool(self, val, _action_id=None):
        with self.handle_action(_action_id):
            match val:
                case ValueHolder(value=CValue(cval=val)):
                    return self.wrap(self.cval(bool(val)))
                case ValueHolder(value=lib.builtins.bool()):
                    return val
                case ValueHolder(value=lib.builtins.object()):
                    return self.wrap(
                        self.call_builtin("scalar.bool.from_python", lib.builtins.bool.__hipy_create_type__(), [val]))
                case _:
                    if hasattr(val.value, "__bool__"):
                        return self.perform_call(self.get_attr(val, "__bool__"))
                    elif hasattr(val.value, "__len__"):
                        return self.perform_binop(self.perform_call(self.get_attr(val, "__len__")),self.constant(0),"__ne__","__ne__")
                    else:
                        return self.wrap(self.cval(True))

    def bool_and(self, left, right, _action_id=None):
        with self.handle_action(_action_id):
            match left, right:
                case (ValueHolder(value=CValue(cval=l)), ValueHolder(value=CValue(cval=r))):
                    return self.wrap(self.cval(l and r))
                case (ValueHolder(value=lib.builtins.bool()), ValueHolder(value=lib.builtins.bool())):
                    return self.wrap(
                        self.call_builtin("scalar.bool.and", lib.builtins.bool.__hipy_create_type__(), [left, right]))
                case _:
                    return self.bool_and(self._to_bool(left), self._to_bool(right))

    def bool_or(self, left, right, _action_id=None):
        with self.handle_action(_action_id):
            match left, right:
                case (ValueHolder(value=CValue(cval=l)), ValueHolder(value=CValue(cval=r))):
                    return self.wrap(self.cval(l or r))
                case (ValueHolder(value=lib.builtins.bool()), ValueHolder(value=lib.builtins.bool())):
                    return self.wrap(
                        self.call_builtin("scalar.bool.or", lib.builtins.bool.__hipy_create_type__(), [left, right]))
                case _:
                    return self.bool_or(self._to_bool(left), self._to_bool(right))

    def bool_not(self, arg, _action_id=None):
        with self.handle_action(_action_id):
            match arg:
                case ValueHolder(value=CValue(cval=val)):
                    return self.wrap(self.cval(not val))
                case ValueHolder(value=lib.builtins.bool()):
                    return self.wrap(
                        self.call_builtin("scalar.bool.not", lib.builtins.bool.__hipy_create_type__(), [arg]))
                case _:
                    return self.bool_not(self._to_bool(arg))

    def use_block(self, block):
        class BlockSwitcher:
            def __init__(self, context, other_block):
                self.context = context
                self.other_block = other_block
                self.old_block = context.block

            def __enter__(self):
                self.context.block = self.other_block

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.context.block = self.old_block

        return BlockSwitcher(self, block)

    def ignore_uses(self, val):
        class Ignore:
            def __init__(self, context, val):
                self.context = context
                self.val = val

            def __enter__(self):
                self.context.ignored_uses.append(self.val)

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.context.ignored_uses.pop()

        return Ignore(self, val)

    def track_using(self, val):
        if val is None:
            return
        assert isinstance(val, ValueHolder)
        if isinstance(val.value, lib.builtins.object):
            return
        if val in self.ignored_uses:
            return
        self.events.append(ValUsage(val))

    def track_same(self, val1, val2):
        assert isinstance(val1, ValueHolder)
        assert isinstance(val2, ValueHolder)
        self.events.append(ValueAlias(val1, val2))

    def track_nested(self, nested, container):
        if isinstance(nested.value, lib.builtins.object):
            return
        assert isinstance(nested, ValueHolder)
        assert isinstance(container, ValueHolder)

        self.events.append(NestedValueAlias(nested, container))

    def track_converted_to_python(self, val):
        assert isinstance(val, ValueHolder)
        self.events.append(ConvertedToPython(val))

    def is_mutable(self, val):
        match val:
            case ValueHolder(value=v):
                match v:
                    case LambdaValue():
                        # actually this is required to retranslate lambdas to python functions
                        return False
                    case _:
                        return type(v).__HIPY_MUTABLE__
        return True

    def to_python(self, val, _action_id=None):
        with self.handle_action(_action_id):
            match val:
                case ValueHolder(value=lib.builtins.object()):
                    return val
            try:

                with self.no_fallback():
                    with self.ignore_uses(val):

                        res = self.perform_call(self.get_attr(val, "__topython__"), [])
                    if self.is_mutable(val):
                        self.track_converted_to_python(val)
                        # todo: track movement of value accross value holders?
                        val.value = res.value
                        return val
                    else:
                        return res
            except (NotImplementedError, TypeError, AttributeError) as e:
                raise e

    def merge(self, val1, val2, fn_val1, fn_val2):
        original_val1 = val1
        original_val2 = val2

        try:
            try:
                return type(val1.value).__merge__(val1, val2, fn_val1, fn_val2, self)
            except (NotImplementedError, AttributeError):
                x,y,fn=type(val2.value).__merge__(val2, val1, fn_val2, fn_val1, self)
                return y,x,fn
        except (NotImplementedError, AttributeError):
            #print("falling back to python when merging values", file=sys.stderr)
            return fn_val1(lambda c: c.to_python(original_val1)), fn_val2(
                lambda c: c.to_python(original_val2)), lambda val: lib.builtins.object(val)

    def _if(self, cond, bodyfn, elsefn, inputs, _action_id):
        #print("if", self)
        # todo: we also need to handle cases where we modify input values
        # the only way: we also need to return the input values
        with self.handle_action(_action_id):
            cond_val = cond
            cond = self.unwrap(cond)
            res = None
            match cond:
                case CValue(cval=val):
                    if val:
                        return bodyfn(*inputs, _context=self)
                    else:
                        return elsefn(*inputs, _context=self)
                case _:
                    cond_val=self._to_bool(cond_val)
                    if_block = ir.Block()
                    else_block = ir.Block()
                    if_early_return = (False, None)
                    else_early_return = (False, None)
                    with self.transaction() as if_transaction:
                        with self.use_block(if_block):
                            try:
                                with self.handle_action("if"):
                                    if_res = bodyfn(*inputs, _context=self)
                            except self.EarlyReturn as e:
                                if_early_return = (True, e.val)
                    if_transaction.commit()
                    if_changed_vals = if_transaction.undo_value_changes()
                    with self.transaction() as else_transaction:
                        with self.use_block(else_block):
                            try:
                                with self.handle_action("else"):
                                    else_res = elsefn(*inputs, _context=self)
                            except self.EarlyReturn as e:
                                else_early_return = (True, e.val)
                    else_transaction.commit()
                    else_changed_vals = else_transaction.undo_value_changes()
                    if if_early_return[0] and else_early_return[0]:
                        if_res = (if_early_return[1],)
                        else_res = (else_early_return[1],)
                    elif if_early_return[0] or else_early_return[0]:
                        raise NotImplementedError("early return in only one branch")
                    if_res_upper = []
                    else_res_upper = []
                    create_merged_fns = []

                    def at_if_block(fn):
                        with self.use_block(if_block):
                            return fn(self)

                    def at_else_block(fn):
                        with self.use_block(else_block):
                            return fn(self)

                    if_i_res_upper = []
                    else_i_res_upper = []
                    create_i_merged_fns = []
                    updated_inputs=[]
                    #print("if_block")
                    #print(if_block)
                    # todo: we also need to handle cases where we modify nested values (e.g. input[0].x)
                    #
                    all_changed=list(set(if_changed_vals.keys()) | set(else_changed_vals.keys()))

                    for c in all_changed:
                        o = if_changed_vals.get(c, else_changed_vals.get(c))[0]
                        if o.t_location in if_changed_vals:
                            i_value = if_changed_vals.get(o.t_location)[1]
                        else:
                            i_value = o.value
                        if o.t_location in else_changed_vals:
                            e_value = else_changed_vals.get(o.t_location)[1]
                        else:
                            e_value = o.value
                        i_type = i_value.__hipy_get_type__()
                        e_type = e_value.__hipy_get_type__()
                        o_type = o.value.__hipy_get_type__()
                        if_i_res_val = self.wrap(i_value)
                        else_i_res_val = self.wrap(e_value)
                        self.track_same(if_i_res_val, o)
                        self.track_same(else_i_res_val, o)
                        if i_type != o_type or e_type != o_type:
                            if_i_res_val, else_i_res_val, create_merged_fn = self.merge(if_i_res_val, else_i_res_val,
                                                                                        at_if_block,
                                                                                        at_else_block)
                        else:
                            create_merged_fn = lambda val: i_type.construct(val, self)
                        if_res =tuple([if_i_res_val if val.t_location == c else val for val in if_res])
                        else_res =tuple([else_i_res_val if val.t_location == c else val for val in else_res])
                        if_i_res_upper.append(if_i_res_val)
                        else_i_res_upper.append(else_i_res_val)
                        create_i_merged_fns.append(create_merged_fn)
                        updated_inputs.append(o)
                    #print("if_block")
                    #print(if_block)
                    for if_res_val, else_res_val in zip(if_res, else_res):
                        if_res_val, else_res_val, create_merged_fn = self.merge(if_res_val, else_res_val, at_if_block,
                                                                                at_else_block)
                        assert isinstance(if_res_val, ValueHolder)
                        assert isinstance(else_res_val, ValueHolder)
                        if_res_upper.append(if_res_val)
                        else_res_upper.append(else_res_val)
                        create_merged_fns.append(create_merged_fn)
                    #print("if_block")
                    #print(if_block)
                    with self.use_block(if_block):
                        if_res_vals = [val.get_ir_value(self) for val in if_res_upper + if_i_res_upper if val is not None]
                        ir.Yield(self.block, if_res_vals)
                    with self.use_block(else_block):
                        else_res_vals = [val.get_ir_value(self) for val in else_res_upper + else_i_res_upper if
                                         val is not None]
                        ir.Yield(self.block, else_res_vals)
                    ifelse = ir.IfElse(self.block, cond_val.get_ir_value(self), [v.type for v in if_res_vals])
                    ifelse.ifBody = if_block
                    ifelse.elseBody = else_block
                    res = []
                    curr_val_idx = 0
                    #print("if_block")
                    #print(if_block)
                    for create_merged_fn, iv in zip(create_merged_fns, if_res_upper):
                        if iv is None:
                            res.append(self.wrap(create_merged_fn(None)))
                        else:
                            res.append(self.wrap(create_merged_fn(ifelse.results[curr_val_idx])))
                            curr_val_idx += 1
                    for create_merged_fn, iv,o in zip(create_i_merged_fns, if_i_res_upper,updated_inputs):
                        r=create_merged_fn(ifelse.results[curr_val_idx])
                        assert isinstance(r,Value)
                        o.value=r
                        curr_val_idx += 1
                    if if_early_return[0] and else_early_return[0]:
                        raise self.EarlyReturn(res[0])
                    return tuple(res)

    def transaction(self):
        class Transaction:
            def __init__(self, context):
                self.context = context
                self.aborted=False
                self.commited=False
                self.id=context.transaction_cntr
                context.transaction_cntr+=1
                self.undo_log= {}

            def __enter__(self):
                self.context.active_transactions.append(self)
                self.original_module = self.context.module
                self.context.module = ir.Module()
                self.original_block = self.context.block
                self.context.block = ir.Block()
                self.original_events = self.context.events
                self.context.events = []
                if self.context.debug:
                    self.original_seen = self.context.seen
                    self.context.seen = copy.deepcopy(self.context.seen)
                self.context.location.append(f"transaction:{self.id}")
                self.context.location.append(None)
                self.context.dyn_actions.append(DynAction())
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.context.active_transactions.pop()
                self.transaction_module = self.context.module
                self.context.module = self.original_module
                self.transaction_block = self.context.block
                self.context.block = self.original_block
                self.transaction_events = self.context.events
                self.context.events = self.original_events
                if self.context.debug:
                    self.transaction_seen = self.context.seen
                    self.context.seen = self.original_seen
                self.context.location.pop()
                self.context.location.pop()
                self.context.dyn_actions.pop()


            def log_set_value(self, val_holder):
                if self.context.ignore_change:
                    return
                if f"transaction:{self.id}" not in val_holder.t_location:
                    if val_holder.t_location not in self.undo_log:
                        #print("transaction", self.id, "logging change", val_holder.t_location, file=sys.stderr)
                        self.undo_log[val_holder.t_location] = (val_holder,copy.deepcopy(val_holder.value))
                pass

            def commit(self):
                self.context.module.merge(self.transaction_module)
                self.context.block.ops.extend(self.transaction_block.ops)
                self.context.events.extend(self.transaction_events)
                if self.context.debug:
                    self.context.seen.update(self.transaction_seen)
                self.propagate_undo_log()
                self.commited=True

            def propagate_undo_log(self):
                if len(self.context.active_transactions)>0:
                    t = self.context.active_transactions[-1]
                    for val_holder, old_val in self.undo_log.values():
                        if f"transaction:{t.id}" not in val_holder.t_location:
                            if val_holder.t_location not in t.undo_log:
                                t.undo_log[val_holder.t_location] = (val_holder, old_val)
            def abort(self):
                self.undo_value_changes()
                self.aborted=True

            def undo_value_changes(self):
                res ={}
                for val_holder,old_val in self.undo_log.values():
                    res[val_holder.t_location]=(val_holder,val_holder.value)
                    self.context.ignore_change=True
                    val_holder.value=old_val
                    self.context.ignore_change=False
                return res

        return Transaction(self)
    def _while(self, cond, body, iter_vals, read_only_inputs, _action_id):
        with self.handle_action(_action_id):
            counter=0
            # we need to make the iter_vals abstract, otherwise, we could rely on constant values that do not hold after the first iteration
            iter_vals=[val.as_abstract(self) for val in iter_vals]
            while True:
                #todo: we also have to account for the case where the provided values are modified.
                #todo: we probably need to register the used values to the transaction in some way?
                with self.transaction() as t:
                    iter_vals_types = [val.value.__hipy_get_type__() for val in iter_vals]
                    with self.handle_action("cond"):
                        cond_val = cond(*(read_only_inputs + iter_vals), _context=self)
                    with self.handle_action("body"):
                        res_vals = body(*(read_only_inputs + iter_vals), _context=self)
                    res_vals_types = [val.value.__hipy_get_type__() for val in res_vals]


                changes=t.undo_value_changes()
                t.abort()
                changed_different_types=False
                for val_holder,after_val in changes.values():
                    before_val=val_holder.value
                    if before_val.__hipy_get_type__() != after_val.__hipy_get_type__():
                        changed_different_types=True
                        after= self.wrap(after_val)
                        new_before_val, after_val, create_merged_fn = self.merge(val_holder, after,
                                                                                 lambda fn: fn(self),
                                                                                 lambda fn: None)
                        val_holder.value=new_before_val.value


                if not changed_different_types and res_vals_types == iter_vals_types:
                    break
                else:
                    for after, before in zip(res_vals, iter_vals):
                        new_before_val, after_val, create_merged_fn = self.merge(before, after,
                                                                                 lambda fn: fn(self),
                                                                                 lambda fn: None)
                        before.value=new_before_val.value

                    if counter > 2:
                        raise RuntimeError("too many iterations")
                    else:
                        counter += 1

            def record_type(vals):
                return ir.RecordType(
                    [(f"val{i}", val.value.__hipy_get_type__().ir_type()) for i, val in enumerate(vals)])
                        # for x, y in zip(iter_vals, iter_vals_args):
            read_only_input_ir_types = record_type(read_only_inputs)
            iter_val_ir_types = record_type(iter_vals)

            func = ir.Function(self.module, f"loop_fn{self.loop_fn_counter}",
                               [read_only_input_ir_types, iter_val_ir_types],
                               record_type(iter_vals))
            condfunc = ir.Function(self.module, f"while_cond_fn{self.loop_fn_counter}",
                               [read_only_input_ir_types, iter_val_ir_types],
                               ir.bool)
            self.loop_fn_counter += 1

            with self.use_block(func.body):
                read_only_inputs_args = [
                    ir.RecordGet(func.body, read_only_inputs[i].value.__hipy_get_type__().ir_type(),
                                 func.args[0],
                                 f"val{i}").result for i in
                    range(len(read_only_inputs))]
                read_only_inputs_args = [
                    self.wrap(o.value.__hipy_get_type__().construct(arg, self)) for
                    o, arg in zip(read_only_inputs, read_only_inputs_args)]
                iter_vals_args = [
                    ir.RecordGet(func.body, iter_vals[i].value.__hipy_get_type__().ir_type(), func.args[1],
                                 f"val{i}").result for i in
                    range(len(iter_vals))]
                iter_vals_args = [self.wrap(o.value.__hipy_get_type__().construct(arg, self))
                                  for
                                  o, arg in zip(iter_vals, iter_vals_args)]
                #print()
                for x, y in zip(iter_vals, iter_vals_args):
                    self.track_same(x, y)
                for x, y in zip(read_only_inputs, read_only_inputs_args):
                    self.track_same(x, y)
                with self.handle_action("body"):
                    res_vals=body(*(read_only_inputs_args + iter_vals_args), _context=self)

                res = ir.MakeRecord(func.body, record_type(res_vals),
                                    {f"val{i}": val.get_ir_value(self) for i, val in enumerate(res_vals)}).result
                ir.Return(func.body, [res])

            with self.use_block(condfunc.body):
                read_only_inputs_args = [
                    ir.RecordGet(self.block, read_only_inputs[i].value.__hipy_get_type__().ir_type(),
                                 condfunc.args[0],
                                 f"val{i}").result for i in
                    range(len(read_only_inputs))]
                read_only_inputs_args = [
                    self.wrap(o.value.__hipy_get_type__().construct(arg, self)) for
                    o, arg in zip(read_only_inputs, read_only_inputs_args)]
                iter_vals_args = [
                    ir.RecordGet(condfunc.body, iter_vals[i].value.__hipy_get_type__().ir_type(), condfunc.args[1],
                                 f"val{i}").result for i in
                    range(len(iter_vals))]
                iter_vals_args = [self.wrap(o.value.__hipy_get_type__().construct(arg, self))
                                  for
                                  o, arg in zip(iter_vals, iter_vals_args)]
                #print()
                for x, y in zip(iter_vals, iter_vals_args):
                    self.track_same(x, y)
                for x, y in zip(read_only_inputs, read_only_inputs_args):
                    self.track_same(x, y)
                with self.handle_action("cond"):
                    res_val=cond(*(read_only_inputs_args + iter_vals_args), _context=self)

                res = self._to_bool(res_val).get_ir_value()
                ir.Return(condfunc.body, [res])

            packed_read_only_inputs = ir.MakeRecord(self.block, record_type(read_only_inputs),
                                                    {f"val{i}": val.get_ir_value(self) for i, val in
                                                     enumerate(read_only_inputs)}).result
            packed_iter_vals = ir.MakeRecord(self.block, record_type(iter_vals),
                                             {f"val{i}": val.get_ir_value(self) for i, val in enumerate(iter_vals)}).result
            func_ref = ir.FunctionRef(self.block, func).result
            cond_func_ref = ir.FunctionRef(self.block, condfunc).result
            res = self.wrap(self.call_builtin("while.iter",RawValue.RawValueType(packed_iter_vals.type),[self.wrap(RawValue(cond_func_ref)),self.wrap(RawValue(func_ref)), self.wrap(RawValue(packed_read_only_inputs)),
                                     self.wrap(RawValue(packed_iter_vals))]))
            res_record = res.value.__value__
            res_vals = [self.wrap(o.value.__hipy_get_type__().construct(
                ir.RecordGet(self.block, o.value.__hipy_get_type__().ir_type(), res_record, f"val{i}").result, self))
                for i, o in enumerate(iter_vals)]
            return tuple(res_vals)
    def _for(self, over, target, body, iter_vals, read_only_inputs, _action_id):
        class NoConstIter(Exception):
            pass
        counter = 0
        with self.handle_action(_action_id):
            try:
                with self.no_fallback():
                    try:
                        const_iter = self.perform_call(self.get_attr(over, "__constiter__"), [])
                    except (NotImplementedError,TypeError, AttributeError) as e:
                        raise NoConstIter()
                match const_iter:
                    case ValueHolder(value=ConstIterValue(values=values)):
                        for i, v in enumerate(values):
                            with self.handle_action(f"const_for{i}"):
                                iter_vals = body(*(read_only_inputs + list(iter_vals) + [v]), _context=self)
                        return tuple(iter_vals)
            except NoConstIter:
                pass
            except (NotImplementedError, TypeError, AttributeError) as e:
                raise e

            iter_val = self.perform_call(self.get_attr(over, "__iter__"), [])
            iter_type = self.perform_call(self.get_attr(iter_val, "__itertype__"), [])
            match iter_type:
                case ValueHolder(value=iter_type_val):
                    match iter_type_val:
                        case HLCClassValue(cls=cls):
                            iter_type = cls.__hipy_create_type__()
                        case TypeValue(type=type):
                            iter_type = type
                        case _:
                            raise RuntimeError(f"Invalid iter_type {iter_type_val}")
            match iter_val:
                case ValueHolder(value=lib.builtins.object()):
                    raise RuntimeError("cannot iterate over python object for now")
            # we need to make the iter_vals abstract, otherwise, we could rely on constant values that do not hold after the first iteration
            iter_vals=[val.as_abstract(self) for val in iter_vals]
            while True:
                #todo: we also have to account for the case where the provided values are modified.
                #todo: we probably need to register the used values to the transaction in some way?
                with self.transaction() as t:
                    iter_vals_types = [val.value.__hipy_get_type__() for val in iter_vals]


                    iter_arg=self.wrap(iter_type.construct(ir.SSAValue(iter_type.ir_type(),None),self))

                    #print(iter_type)
                    res_vals = body(*(read_only_inputs + iter_vals + [iter_arg]), _context=self)
                    res_vals_types = [val.value.__hipy_get_type__() for val in res_vals]


                changes=t.undo_value_changes()
                t.abort()
                changed_different_types=False
                for val_holder,after_val in changes.values():
                    before_val=val_holder.value
                    if before_val.__hipy_get_type__() != after_val.__hipy_get_type__():
                        changed_different_types=True
                        after= self.wrap(after_val)
                        new_before_val, after_val, create_merged_fn = self.merge(val_holder, after,
                                                                                 lambda fn: fn(self),
                                                                                 lambda fn: None)
                        val_holder.value=new_before_val.value


                if not changed_different_types and res_vals_types == iter_vals_types:
                    break
                else:
                    for after, before in zip(res_vals, iter_vals):
                        new_before_val, after_val, create_merged_fn = self.merge(before, after,
                                                                                 lambda fn: fn(self),
                                                                                 lambda fn: None)
                        before.value=new_before_val.value

                    if counter > 2:
                        raise RuntimeError("too many iterations")
                    else:
                        counter += 1

            def record_type(vals):
                return ir.RecordType(
                    [(f"val{i}", val.value.__hipy_get_type__().ir_type()) for i, val in enumerate(vals)])
                        # for x, y in zip(iter_vals, iter_vals_args):
            read_only_input_ir_types = record_type(read_only_inputs)
            iter_val_ir_types = record_type(iter_vals)

            func = ir.Function(self.module, f"loop_fn{self.loop_fn_counter}",
                               [read_only_input_ir_types, iter_val_ir_types, iter_type.ir_type()],
                               record_type(iter_vals))
            self.loop_fn_counter += 1

            with self.use_block(func.body):
                read_only_inputs_args = [
                    ir.RecordGet(func.body, read_only_inputs[i].value.__hipy_get_type__().ir_type(),
                                 func.args[0],
                                 f"val{i}").result for i in
                    range(len(read_only_inputs))]
                read_only_inputs_args = [
                    self.wrap(o.value.__hipy_get_type__().construct(arg, self)) for
                    o, arg in zip(read_only_inputs, read_only_inputs_args)]
                iter_vals_args = [
                    ir.RecordGet(func.body, iter_vals[i].value.__hipy_get_type__().ir_type(), func.args[1],
                                 f"val{i}").result for i in
                    range(len(iter_vals))]
                iter_vals_args = [self.wrap(o.value.__hipy_get_type__().construct(arg, self))
                                  for
                                  o, arg in zip(iter_vals, iter_vals_args)]
                #print()
                iter_arg = self.wrap(iter_type.construct(func.args[2], self))
                for x, y in zip(iter_vals, iter_vals_args):
                    self.track_same(x, y)
                for x, y in zip(read_only_inputs, read_only_inputs_args):
                    self.track_same(x, y)
                iter_val.value.__track__(iter_arg, self)
                res_vals=body(*(read_only_inputs_args + iter_vals_args + [iter_arg]), _context=self)

                res = ir.MakeRecord(func.body, record_type(res_vals),
                                    {f"val{i}": val.get_ir_value(self) for i, val in enumerate(res_vals)}).result
                ir.Return(func.body, [res])

            packed_read_only_inputs = ir.MakeRecord(self.block, record_type(read_only_inputs),
                                                    {f"val{i}": val.get_ir_value(self) for i, val in
                                                     enumerate(read_only_inputs)}).result
            packed_iter_vals = ir.MakeRecord(self.block, record_type(iter_vals),
                                             {f"val{i}": val.get_ir_value(self) for i, val in enumerate(iter_vals)}).result
            func_ref = ir.FunctionRef(self.block, func).result
            res = self.perform_call(self.get_attr(iter_val, "__iterate__"),
                                    [self.wrap(RawValue(func_ref)), self.wrap(RawValue(packed_read_only_inputs)),
                                     self.wrap(RawValue(packed_iter_vals))])
            res_record = res.value.__value__
            res_vals = [self.wrap(o.value.__hipy_get_type__().construct(
                ir.RecordGet(self.block, o.value.__hipy_get_type__().ir_type(), res_record, f"val{i}").result, self))
                for i, o in enumerate(iter_vals)]
            return tuple(res_vals)
        # print(res_vals)

    def create_lambda(self, staged, bind_python, bind_staged, _action_id):
        with self.handle_action(_action_id):
            return self.wrap(LambdaValue(staged, bind_python, bind_staged))

    def create_tuple(self, elts, _action_id=None):
        with self.handle_action(_action_id):
            return self.wrap(lib.builtins.tuple(tuple(elts)))

    def create_list(self, elts, _action_id=None):
        with self.handle_action(_action_id):
            res = self.wrap(lib.builtins._concrete_list(elts))
            for l in elts:
                self.track_nested(l, res)
            return res
    def create_dict_simple(self, d, _action_id=None):
        with self.handle_action(_action_id):
            return self.create_dict([self.constant(k) for k in d], [v for v in d.values()])
    def create_dict(self, keys, values, _action_id=None):
        with self.handle_action(_action_id):
            assert all([isinstance(val, ValueHolder) for val in values])
            if all([isinstance(key.value, CValue) for key in keys]):
                return self.wrap(
                    lib.builtins._concrete_dict({key.value.cval: value for key, value in zip(keys, values)}))
            else:
                raise RuntimeError("cannot create dict with non-constant keys for now")

    def create_slice(self, start, stop, step, _action_id=None):
        with self.handle_action(_action_id):
            return self.wrap(lib.builtins.slice(start, stop, step))

    def make_record(self, names, values):
        raw_values = [value.get_ir_value(self) for value in values]
        record_type = ir.RecordType([(n, v.type) for n, v in zip(names, raw_values)])

        return ir.MakeRecord(self.block, record_type, {n: v for n, v in zip(names, raw_values)}).result

    def _try(self, tryfn, exceptfn, try_closure, except_closure, _action_id=None):
        with self.handle_action(_action_id):
            self.try_cntr+=1
            try_counter = self.try_cntr
            try_early_return = (False, None)
            except_early_return = (False, None)
            try_closure_obj = binding.create_closure(try_closure, self)
            except_closure_obj = binding.create_closure(except_closure, self)
            try_closure_type = try_closure_obj.value.__hipy_get_type__()
            try_explict_closure_arg = not isinstance(try_closure_type.ir_type(), ir.VoidType)
            try_func = ir.Function(self.module, f"try_fn{try_counter}",
                                   [try_closure_type.ir_type()] if try_explict_closure_arg else [], ir.void)
            with self.use_block(try_func.body):
                closure_param = try_func.args[-1] if try_explict_closure_arg else ir.Constant(self.block, 0, ir.void).result
                closure_param = self.wrap(try_closure_type.construct(closure_param, self))
                try:
                    with self.handle_action("try"):
                        try_results = tryfn(_context=self, __closure__=closure_param)
                except self.EarlyReturn as e:
                    try_early_return = (True, e.val)
            except_closure_type = except_closure_obj.value.__hipy_get_type__()
            except_explict_closure_arg = not isinstance(except_closure_type.ir_type(), ir.VoidType)
            except_func = ir.Function(self.module, f"except_fn{try_counter}",
                                      [except_closure_type.ir_type()] if except_explict_closure_arg else [], ir.void)
            with self.use_block(except_func.body):
                closure_param = except_func.args[-1] if except_explict_closure_arg else ir.Constant(self.block, 0, ir.void).result
                closure_param = self.wrap(except_closure_type.construct(closure_param, self))
                try:
                    with self.handle_action("except"):
                        except_results = exceptfn(_context=self, __closure__=closure_param)
                except self.EarlyReturn as e:
                    except_early_return = (True, e.val)
            if try_early_return[0] and except_early_return[0]:
                try_results = (try_early_return[1],)
                except_results = (except_early_return[1],)
            elif try_early_return[0] or except_early_return[0]:
                raise NotImplementedError("early return in only one branch")
            try_res_upper = []
            except_res_upper = []
            create_merged_fns = []

            def at_try_block(fn):
                with self.use_block(try_func.body):
                    return fn(self)

            def at_except_block(fn):
                with self.use_block(except_func.body):
                    return fn(self)

            for try_res_val, except_res_val in zip(try_results, except_results):
                try_res_val, except_res_val, create_merged_fn = self.merge(try_res_val, except_res_val, at_try_block,
                                                                           at_except_block)
                assert isinstance(try_res_val, ValueHolder)
                assert isinstance(except_res_val, ValueHolder)
                try_res_upper.append(try_res_val)
                except_res_upper.append(except_res_val)
                create_merged_fns.append(create_merged_fn)

            with self.use_block(try_func.body):
                try_res_vals = [val.get_ir_value(self) for val in try_res_upper if val is not None]
                record_type = ir.RecordType([(f"r{i}", val.type) for i, val in enumerate(try_res_vals)])
                record = ir.MakeRecord(self.block, record_type,
                                       {f"r{i}": val for i, val in enumerate(try_res_vals)}).result
                try_func.res_type = record.type
                ir.Return(self.block, [record])
            with self.use_block(except_func.body):
                except_res_vals = [val.get_ir_value(self) for val in except_res_upper if
                                   val is not None]
                record_type = ir.RecordType([(f"r{i}", val.type) for i, val in enumerate(except_res_vals)])
                record = ir.MakeRecord(self.block, record_type,
                                       {f"r{i}": val for i, val in enumerate(except_res_vals)}).result
                except_func.res_type = record.type
                ir.Return(self.block, [record])

            try_ref = ir.FunctionRef(self.block, try_func,
                                     try_closure_obj.get_ir_value() if try_explict_closure_arg else None).result
            except_ref = ir.FunctionRef(self.block, except_func,
                                        except_closure_obj.get_ir_value() if except_explict_closure_arg else None).result
            res_record=ir.CallBuiltin(self.block, "try_except", [try_ref, except_ref], try_func.res_type).result

            curr_val_idx = 0
            res=[]
            for create_merged_fn, iv in zip(create_merged_fns, try_res_upper):
                if iv is None:
                    res.append(self.wrap(create_merged_fn(None)))
                else:
                    res.append(self.wrap(create_merged_fn(ir.RecordGet(self.block, try_res_vals[curr_val_idx].type, res_record, f"r{curr_val_idx}").result)))
                    curr_val_idx += 1

            if try_early_return[0] and except_early_return[0]:
                raise self.EarlyReturn(res[0])
            return tuple(res)

    def get_corresponding_class(self, cls):
        if cls.__module__ in mocked_modules:
            if hasattr(mocked_modules[cls.__module__],cls.__name__):
                return getattr(mocked_modules[cls.__module__],cls.__name__)
        return None

    def __deepcopy__(self, memo):
        # Return self instead of creating a deepcopy
        return self