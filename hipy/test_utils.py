import hipy
from hipy.value import ValueHolder


@hipy.raw
def not_constant(value, _context):
    return value.as_abstract(_context)