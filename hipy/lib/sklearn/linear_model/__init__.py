__HIPY_MODULE__ = "sklearn.linear_model"
import sys

import hipy
hipy.register(sys.modules[__name__])
from ._base import LinearRegression
from ._logistic import LogisticRegression
