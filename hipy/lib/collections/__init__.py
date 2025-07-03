__HIPY_MODULE__ = "collections"

from typing import List

import hipy
import sys

from hipy import intrinsics, ir
from hipy.internal_values import _named_tuple
from hipy.lib.builtins import tuple as _tuple, _concrete_list
from hipy.value import Type, ValueHolder, CValue, HLCClassValue, raw_module

hipy.register(sys.modules[__name__])

import collections
@hipy.raw
def namedtuple(typename, field_names, _context):
    match typename:
        case ValueHolder(value=CValue(cval=typename)):
            pass
        case _:
            raise RuntimeError(f"Invalid argument for typename {typename}")
    _field_names = []
    match field_names:
        case ValueHolder(value=_concrete_list(items=field_names)):
            pass
        case _:
            raise RuntimeError(f"Invalid argument for field names {field_names}")
    for name in field_names:
        match name:
            case ValueHolder(value=CValue(cval=name)):
                _field_names.append(name)
            case _:
                raise RuntimeError(f"Invalid argument for field name {name}")

    # intrinsics.only_implemented_if(rename==False, defaults is None, module is None)
    @hipy.classdef
    class _specialized_named_tuple(_named_tuple):
        def __init__(self, elts, value=None):
            super().__init__(elts, typename, _field_names, value=value)

        @staticmethod
        @hipy.raw
        def __create__(*args, _context=None, **kwargs):
            d = {f:a for f,a in zip(_field_names, args)}
            d.update(kwargs)
            l = [d[f] for f in _field_names]
            return _context.wrap(_specialized_named_tuple(l))

        def __hipy_create_type__(*args) -> Type:
            return _named_tuple.NamedTupleType(typename, _field_names, [x.__hipy_get_type__() for x in args],
                                               cls=_specialized_named_tuple)

        def __hipy_get_type__(self):
            return _named_tuple.NamedTupleType(typename, _field_names, [x.value.__hipy_get_type__() for x in self._elts],
                                               cls=_specialized_named_tuple)

        @hipy.raw
        def __topython__(self):
            namedtuplefn=_context.get_attr(_context.import_pymodule("collections"), "namedtuple")
            namedtuplecls= _context.perform_call(namedtuplefn, [_context.constant(self.value._typename), _context.create_list([_context.constant(f) for f in self.value._field_names])])
            return _context.perform_call(namedtuplecls, self.value._elts)


    return _context.wrap(HLCClassValue(_specialized_named_tuple))
