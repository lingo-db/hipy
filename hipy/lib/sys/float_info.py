import sys as _pysys

# Exposed as a shim sub-module of ``hipy.lib.sys`` so compiled code
# reaches ``sys.float_info.max`` through the module-attribute path.
max = _pysys.float_info.max
min = _pysys.float_info.min
epsilon = _pysys.float_info.epsilon
