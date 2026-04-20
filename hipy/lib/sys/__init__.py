import sys as _pysys

import hipy
from hipy.lib.sys import float_info

__HIPY_MODULE__ = "sys"

hipy.register(_pysys.modules[__name__])
