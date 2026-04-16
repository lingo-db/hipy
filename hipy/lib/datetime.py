__HIPY_MODULE__ = "datetime"

import hipy
import sys

from hipy import intrinsics, ir
from hipy.value import ValueHolder, raw_module, Value, SimpleType

hipy.register(sys.modules[__name__])


@hipy.classdef
class date(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)

    @staticmethod
    @hipy.compiled_function
    def __create__(year, month, day):
        if intrinsics.isa(year, int) and intrinsics.isa(month, int) and intrinsics.isa(day, int):
            return intrinsics.call_builtin("date.from_ymd", date, [year, month, day])
        else:
            intrinsics.not_implemented()
            return intrinsics.undef(date)

    @staticmethod
    @hipy.compiled_function
    def today():
        return intrinsics.call_builtin("date.today", date, [])

    @staticmethod
    @hipy.compiled_function
    def fromisoformat(s):
        if intrinsics.isa(s, str):
            return intrinsics.call_builtin("date.from_isoformat", date, [s])
        else:
            intrinsics.not_implemented()
            return intrinsics.undef(date)

    @hipy.compiled_function
    def __topython__(self):
        py_datetime = intrinsics.import_pymodule("datetime")
        py_date_cls = intrinsics.get_attr(py_datetime, "date")
        return py_date_cls(self.year, self.month, self.day)

    @hipy.compiled_function
    def __str__(self):
        return intrinsics.call_builtin("date.to_string", str, [self], side_effects=False)

    @hipy.compiled_function
    def __hipy__repr__(self):
        return "datetime.date(" + str(self.year) + ", " + str(self.month) + ", " + str(self.day) + ")"

    @hipy.compiled_function
    def __sub__(self, other):
        if intrinsics.isa(other, date):
            return intrinsics.call_builtin("date.diff", timedelta, [self, other], side_effects=False)
        elif intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("date.sub_interval", date, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return self

    @hipy.compiled_function
    def __add__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("date.add_interval", date, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return self

    @hipy.compiled_function
    def __radd__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("date.add_interval", date, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return self

    @hipy.compiled_function
    def __eq__(self, other):
        if intrinsics.isa(other, date):
            return intrinsics.call_builtin("date.compare.eq", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __ne__(self, other):
        if intrinsics.isa(other, date):
            return intrinsics.call_builtin("date.compare.neq", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __lt__(self, other):
        if intrinsics.isa(other, date):
            return intrinsics.call_builtin("date.compare.lt", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __le__(self, other):
        if intrinsics.isa(other, date):
            return intrinsics.call_builtin("date.compare.lte", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __gt__(self, other):
        if intrinsics.isa(other, date):
            return intrinsics.call_builtin("date.compare.gt", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __ge__(self, other):
        if intrinsics.isa(other, date):
            return intrinsics.call_builtin("date.compare.gte", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __hipy_getattr__(self, item):
        if item == "year":
            return intrinsics.call_builtin("date.get_year", int, [self], side_effects=False)
        elif item == "month":
            return intrinsics.call_builtin("date.get_month", int, [self], side_effects=False)
        elif item == "day":
            return intrinsics.call_builtin("date.get_day", int, [self], side_effects=False)
        else:
            intrinsics.not_implemented()
            return 0

    @hipy.compiled_function
    def weekday(self):
        return intrinsics.call_builtin("date.weekday", int, [self], side_effects=False)

    @hipy.compiled_function
    def isoweekday(self):
        return intrinsics.call_builtin("date.isoweekday", int, [self], side_effects=False)

    @hipy.compiled_function
    def isoformat(self):
        return intrinsics.call_builtin("date.to_string", str, [self], side_effects=False)

    @hipy.compiled_function
    def toordinal(self):
        return intrinsics.call_builtin("date.toordinal", int, [self], side_effects=False)

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, date):
            return self, other, lambda val: date(val)
        else:
            raise NotImplementedError()

    @staticmethod
    def __hipy_create_type__(*args):
        return SimpleType(date, ir.date)

    def __hipy_get_type__(self):
        return SimpleType(date, ir.date)


@hipy.classdef
class timedelta(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value):
        super().__init__(value)

    @staticmethod
    @hipy.compiled_function
    def __create__(days=0, seconds=0):
        if intrinsics.isa(days, int) and intrinsics.isa(seconds, int):
            return intrinsics.call_builtin("interval.from_days_seconds", timedelta, [days, seconds], side_effects=False)
        else:
            intrinsics.not_implemented()
            return intrinsics.undef(timedelta)

    @hipy.compiled_function
    def __topython__(self):
        py_datetime = intrinsics.import_pymodule("datetime")
        py_timedelta_cls = intrinsics.get_attr(py_datetime, "timedelta")
        return py_timedelta_cls(self.days, self.seconds)

    @hipy.compiled_function
    def __str__(self):
        return intrinsics.call_builtin("interval.to_string", str, [self], side_effects=False)

    @hipy.compiled_function
    def __hipy__repr__(self):
        return "datetime.timedelta(days=" + str(self.days) + ", seconds=" + str(self.seconds) + ")"

    @hipy.compiled_function
    def __add__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("interval.add", timedelta, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return self

    @hipy.compiled_function
    def __sub__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("interval.sub", timedelta, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return self

    @hipy.compiled_function
    def __neg__(self):
        return intrinsics.call_builtin("interval.neg", timedelta, [self], side_effects=False)

    @hipy.compiled_function
    def __eq__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("interval.compare.eq", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __ne__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("interval.compare.neq", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __lt__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("interval.compare.lt", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __le__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("interval.compare.lte", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __gt__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("interval.compare.gt", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __ge__(self, other):
        if intrinsics.isa(other, timedelta):
            return intrinsics.call_builtin("interval.compare.gte", bool, [self, other], side_effects=False)
        else:
            intrinsics.not_implemented()
            return False

    @hipy.compiled_function
    def __bool__(self):
        return intrinsics.call_builtin("interval.is_nonzero", bool, [self], side_effects=False)

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        if isinstance(other.value, timedelta):
            return self, other, lambda val: timedelta(val)
        else:
            raise NotImplementedError()

    @staticmethod
    def __hipy_create_type__(*args):
        return SimpleType(timedelta, ir.interval)

    def __hipy_get_type__(self):
        return SimpleType(timedelta, ir.interval)

    @hipy.compiled_function
    def __hipy_getattr__(self, item):
        if item == "days":
            return intrinsics.call_builtin("interval.days", int, [self], side_effects=False)
        elif item == "seconds":
            return intrinsics.call_builtin("interval.seconds", int, [self], side_effects=False)
        else:
            intrinsics.not_implemented()
            return 0

    @hipy.compiled_function
    def total_seconds(self):
        return intrinsics.call_builtin("interval.total_seconds", float, [self], side_effects=False)
