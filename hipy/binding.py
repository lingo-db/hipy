import hipy
from hipy import ir
from hipy.value import Value, Type, CValue, ValueHolder, MaterializedConstantValue, LambdaValue


@hipy.classdef
class closure_record(Value):

    def __init__(self, elts, field_names, is_void=None, value=None):
        super().__init__(value)
        if is_void is None:
            self.is_void = {}
        else:
            self.is_void = is_void
        self.values = {}
        for e, n in zip(elts, field_names):
            if isinstance(e.value.__hipy_get_type__().ir_type(), ir.VoidType):
                self.is_void[n] = e.value.__hipy_get_type__()
            else:
                self.values[n] = e

    class closure_record_type(Type):
        def __init__(self, field_types, field_names, is_void):
            self._field_types = field_types
            self._field_names = field_names
            self._is_void = is_void

        def ir_type(self):
            if len(self._field_types) == 0:
                return ir.void
            elif len(self._field_types) == 1:
                return self._field_types[0].ir_type()
            else:
                assert not any([isinstance(t.ir_type(), ir.VoidType) for t in self._field_types])
                return ir.RecordType([(n, t.ir_type()) for n, t in zip(self._field_names, self._field_types)])

        def construct(self, value, context):
            if len(self._field_types) == 0:
                return closure_record([], [], self._is_void, value=value)
            elif len(self._field_types) == 1:
                return closure_record([context.wrap(self._field_types[0].construct(value, context))],
                                      [self._field_names[0]], self._is_void, value=value)
            else:
                return closure_record(
                    [context.wrap(t.construct(ir.RecordGet(context.block, t.ir_type(), value, n).result, context)) for
                     t, n in
                     zip(self._field_types, self._field_names)], self._field_names, self._is_void, value=value)

        def __eq__(self, other):
            if isinstance(other, closure_record.closure_record_type):
                return self._field_types == other._field_types and self._field_names == other._field_names and self._is_void == other._is_void
            else:
                return False

    def __hipy_create_type__(*args) -> Type:
        return closure_record.closure_record_type(*args)

    def __hipy_get_type__(self):
        field_names = list(self.values.keys())
        return closure_record.closure_record_type([self.values[n].value.__hipy_get_type__() for n in field_names],
                                                  field_names, self.is_void)

    def __abstract__(self, context):
        type = self.__hipy_get_type__()
        if isinstance(type.ir_type(), ir.VoidType):
            return type.construct(ir.Constant(context.block, 0, ir.void).result, context)
        elif len(self.values) == 1:
            return type.construct(list(self.values.values())[0].get_ir_value(context), context)
        else:
            return type.construct(ir.MakeRecord(context.block, type.ir_type(),
                                                {n: v.get_ir_value(context) for n, v in self.values.items()}).result,
                                  context)

    @hipy.raw
    def __getitem__(self, item, _context):
        match item:
            case ValueHolder(value=CValue(cval=item)):
                if item in self.value.is_void:
                    return _context.wrap(
                        self.value.is_void[item].construct(ir.Constant(_context.block, 0, ir.void).result, _context))
                return self.value.values[item]

    @hipy.raw
    def __topython__(self):
        raise NotImplementedError()


@hipy.classdef
class lambda_in_closure(Value):
    def __init__(self, closureVal, lambda_staged_fn):
        super().__init__(closureVal.get_ir_value())
        self._lambda_staged_fn = lambda_staged_fn
        self._closure_value = closureVal

    class lambda_in_closure_type(Type):
        def __init__(self, closure_type, lambda_staged_fn):
            self._closure_type = closure_type
            self._lambda_staged_fn = lambda_staged_fn

        def ir_type(self):
            return self._closure_type.ir_type()

        def construct(self, value, context):
            return lambda_in_closure(context.wrap(self._closure_type.construct(value, context)),
                                     self._lambda_staged_fn)

        def __eq__(self, other):
            if isinstance(other, lambda_in_closure.lambda_in_closure_type):
                return self._closure_type == other._closure_type and self._lambda_staged_fn == other._lambda_staged_fn
            else:
                return False

        def __repr__(self):
            return f"lambda_in_closure.lambda_in_closure_type({self._closure_type}, {self._lambda_staged_fn})"

    def __hipy_get_type__(self):
        return lambda_in_closure.lambda_in_closure_type(self._closure_value.value.__hipy_get_type__(),
                                                        self._lambda_staged_fn)

    def __hipy_create_type__(self, closure_type, lambda_staged_fn) -> Type:
        return lambda_in_closure.lambda_in_closure_type(closure_type, lambda_staged_fn)

    @hipy.raw
    def __topython__(self):
        raise NotImplementedError()

    @hipy.raw
    def __call__(self, *args, _context, **kwargs):
        return self.value._lambda_staged_fn(*args, **kwargs, __closure__=self.value._closure_value,
                                            _context=_context)


def create_closure(closure_dict,_context):
    def keep_const(vh):
        value = vh.value
        if value.__HIPY_MUTABLE__ or value.__HIPY_MATERIALIZED__ or value.__HIPY_NESTED_OBJECTS__:
            return vh
        else:
            return _context.wrap(MaterializedConstantValue(value))

    def convert_lambda(l: LambdaValue):
        def bind_nested_fn(nested_closure_dict, nested_fn):
            nested_closure_dict = {**nested_closure_dict}
            for k, v in nested_closure_dict.items():
                match v.value:
                    case LambdaValue():
                        nested_closure_dict[k] = convert_lambda(v.value)
                    case _:
                        nested_closure_dict[k] = keep_const(v)
            keys = list(nested_closure_dict.keys())
            if len(keys) == 0:
                return _context.wrap(lambda_in_closure(_context.constant(None), nested_fn))
            closure = _context.wrap(
                closure_record([nested_closure_dict[k] for k in keys], keys))
            return _context.wrap(lambda_in_closure(closure, nested_fn))

        return l.bind_staged(bind_nested_fn)

    closure_dict = {**closure_dict}
    for k, v in closure_dict.items():
        match v.value:
            case LambdaValue():
                closure_dict[k] = convert_lambda(v.value)
            case _:
                closure_dict[k] = keep_const(v)

    keys = list(closure_dict.keys())
    closure = _context.wrap(closure_record([closure_dict[k] for k in keys], keys))
    return closure