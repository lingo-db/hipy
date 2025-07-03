__HIPY_MODULE__ = "pickle"

import hipy
import sys

from hipy.lib.builtins import _const_bytes
from hipy.value import ValueHolder, raw_module

hipy.register(sys.modules[__name__])
import pickle


@hipy.raw
def loads(value, _context):
    match value:
        case ValueHolder(value=_const_bytes(cval=b)):
            python_value= pickle.loads(b)
            corresponding=_context.get_corresponding_class(type(python_value))
            if corresponding is not None and hasattr(corresponding, "__from_constant__"):
                return corresponding.__from_constant__(python_value,_context)
    return _context.constant(None)