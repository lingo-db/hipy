
import hipy
from hipy import intrinsics, ir

@hipy.compiled_function
def execute(types, query, *params):
    if len(types) == 1:
        ret_type = types[0]
        return intrinsics.call_builtin("sql.execute", ret_type, [query, *params])
