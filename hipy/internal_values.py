from typing import List

import hipy
from hipy import ir
from hipy.lib.builtins import tuple as _tuple
from hipy.value import Type, ValueHolder, CValue, HLCFunctionValue


@hipy.classdef
class _named_tuple(_tuple):
    __HIPY_MUTABLE__ = False

    def __init__(self, elts, typename, field_names, value=None):
        self._field_names = field_names
        self._typename = typename
        super().__init__(elts, value)

    class NamedTupleType(Type):
        def __init__(self, typename, field_names, element_types: List[Type], cls=None):
            if cls is None:
                cls = _named_tuple
            self.cls = cls
            self.typename = typename
            self.field_names = field_names
            self.element_types = element_types

        def ir_type(self):
            return ir.RecordType([(f"_elt{i}", t.ir_type()) for i, t in enumerate(self.element_types)])

        def construct(self, value, context):
            args = []
            for i, t in enumerate(self.element_types):
                n = f"_elt{i}"
                args.append(context.wrap(
                    t.construct(ir.RecordGet(context.block, t.ir_type(), value, n).result, context)))
            return self.cls(args, self.typename, self.field_names, value=value)

        def __eq__(self, other):
            if isinstance(other, _named_tuple.NamedTupleType):
                return self.element_types == other.element_types and self.field_names == other.field_names and self.typename == other.typename
            else:
                return False

    def __hipy_get_type__(self):
        return _named_tuple.NamedTupleType(self._typename, self._field_names, [x.value.__hipy_get_type__() for x in self._elts])

    def __hipy_create_type__(*args) -> Type:
        raise NotImplementedError()

    @hipy.raw
    def __hipy_getattr__(self, name, _context):
        match name:
            case ValueHolder(value=CValue(cval=name)):
                pass
            case _:
                raise NotImplementedError()
        if name in self.value._field_names:
            return _context.get_item(self, _context.constant(self.value._field_names.index(name)))
        else:
            raise AttributeError(f"named tuple has no attribute {name}")

    @hipy.raw
    def __hipy__repr__(self, _context):
        @hipy.compiled_function
        def repr_pair(f, e):
            return f+"="+repr(e)
        res = _context.constant(f"{self.value._typename}(")
        first = True
        for f, e in zip(self.value._field_names, self.value._elts):
            if first:
                first = False
            else:
                res = _context.perform_binop(res, _context.constant(", "), "__add__", "__radd__")
            pair = _context.perform_call(_context.wrap(HLCFunctionValue(repr_pair)), [_context.constant(f), e])
            res = _context.perform_binop(res, pair, "__add__", "__radd__")
        res = _context.perform_binop(res, _context.constant(")"), "__add__", "__radd__")
        return res


