
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

    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):
        from hipy.value import VoidValue
        element_type = self.value.element_type
        nullable_type = Nullable.NullableType(element_type)
        construct = lambda val: Nullable(val, element_type)
        if isinstance(other.value, Nullable):
            if other.value.element_type != element_type:
                raise NotImplementedError()
            return self, other, construct
        if isinstance(other.value, VoidValue):
            return self, other_fn(lambda c: c.wrap(c.call_builtin("nullable.null", nullable_type, []))), construct
        try:
            other_type = other.value.__hipy_get_type__()
        except (NotImplementedError, AttributeError):
            raise NotImplementedError()
        if other_type != element_type:
            raise NotImplementedError()
        return self, other_fn(lambda c: c.wrap(c.call_builtin("nullable.make", nullable_type, [other]))), construct

    @hipy.compiled_function
    def is_null(self):
        return intrinsics.call_builtin("nullable.is_null", bool, [self])

    @hipy.compiled_function
    def __is_none__(self):
        # `x is None` / `x is not None`
        return self.is_null()

    @hipy.raw
    def __eq__(self, other, _context):
        # Python semantics: None equals only None, a value is compared with the other operand
        from hipy.value import VoidValue
        if isinstance(other.value, VoidValue):
            return _context.perform_call(_context.get_attr(self, "is_null"))
        return _context.perform_call(_context.get_attr(self, "_eq_value"), [other])

    @hipy.compiled_function
    def _eq_value(self, other):
        if self.is_null():
            return other is None
        else:
            return self.get_value() == other

    @hipy.compiled_function
    def __ne__(self, other):
        return not (self == other)

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


@hipy.compiled_function
def row(*element_types):
    """Type of a row-valued sql.execute: a tuple with one element per result
    column, e.g. sql.row(str, sql.nullable(int)): the values of the query's
    first row (an arbitrary one if there are several). If the query yields no
    row, the query fails with a runtime error."""
    return intrinsics.create_type(tuple, [t for t in element_types])




@hipy.compiled_function
def rows(*element_types):
    """Type of a list-valued sql.execute: a list with one tuple per result row
    (each as for sql.row(...)), in the order of the query (unspecified without
    ORDER BY), e.g. sql.rows(int, sql.nullable(float)). Strings are not
    supported yet (a compile error)."""
    return intrinsics.create_type(list, intrinsics.create_type(tuple, [t for t in element_types]))
