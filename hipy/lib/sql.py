
import hipy
from hipy import intrinsics, ir
from hipy.value import ValueHolder, raw_module, Value, SimpleType, Type


@hipy.compiled_function
def execute(type, query, *params):
    return intrinsics.call_builtin("sql.execute", type, [query]+[p for p in params])



@hipy.classdef
class Nullable(Value):
    __HIPY_MUTABLE__ = False
    __HIPY_NESTED_OBJECTS__ = False

    def __init__(self, value, element_type):
        super().__init__(value)
        self.element_type = element_type


    @hipy.compiled_function
    def __topython__(self):
        return intrinsics.not_implemented()

    @hipy.compiled_function
    def is_null(self):
        return intrinsics.call_builtin("nullable.is_null", bool, [self])

    @hipy.compiled_function
    def get_value(self):
        return intrinsics.call_builtin("nullable.get_value", self.element_type, [self])

    @hipy.compiled_function
    def get_value_or_default(self, default_value):
        if self.is_null():
            return default_value
        else:
            return self.get_value()

    class NullableType(Type):
        def __init__(self, element_type: Type):
            self.element_type = element_type

        def ir_type(self):
            return ir.NullableType(self.element_type.ir_type())

        def construct(self, value, context):
            return Nullable(value, self.element_type)

        def __eq__(self, other):
            if isinstance(other, Nullable.NullableType):
                return self.element_type == other.element_type
            else:
                return False

        def __repr__(self):
            return f"nullable.NullableType({self.element_type})"

        def get_cls(self):
            return Nullable

    @staticmethod
    def __hipy_create_type__(*args):
        return Nullable.NullableType(*args)

    def __hipy_get_type__(self):
        return Nullable.NullableType(self.element_type)

@hipy.compiled_function
def nullable(element_type):
    return intrinsics.create_type(Nullable, element_type)


