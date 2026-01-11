__HIPY_MODULE__ = "datetime"

import hipy
import sys

from hipy import intrinsics, ir
from hipy.lib.builtins import _const_bytes
from hipy.value import ValueHolder, raw_module, Value, SimpleType

hipy.register(sys.modules[__name__])





@hipy.classdef
class date(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)


    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.not_implemented()

    @hipy.compiled_function
    def __sub__(self, other):
        if intrinsics.isa(other, date):
            return intrinsics.call_builtin("date.diff", timedelta, [self, other])
        else:
            return intrinsics.not_implemented()


    @staticmethod
    def __hipy_create_type__(*args):
        return SimpleType(date, ir.date)

    def __hipy_get_type__(self):
        return SimpleType(int, ir.date)



@hipy.classdef
class timedelta(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)


    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.not_implemented()


    @staticmethod
    def __hipy_create_type__(*args):
        return SimpleType(timedelta, ir.interval)

    def __hipy_get_type__(self):
        return SimpleType(timedelta, ir.interval)

    @hipy.compiled_function
    def __hipy_getattr__(self, item):
        if item == "days":
            return intrinsics.call_builtin("interval.days", int, [self])
        else:
            return intrinsics.not_implemented()


