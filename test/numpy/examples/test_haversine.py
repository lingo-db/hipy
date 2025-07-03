import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np

@hipy.compiled_function
def gen_data(size):
    lats = np.ones(size, dtype=np.float64) * 0.0698132
    lons = np.ones(size, dtype=np.float64) * 0.0698132
    return lats, lons


@hipy.compiled_function
def haversine(lat2, lon2):
    miles_constant = 3959.0
    lat1 = 0.70984286
    lon1 = 1.2389197

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    mi = miles_constant * c
    return mi

@hipy.compiled_function
def run_haversine():
    lats, lons = gen_data(10)
    print(haversine(lats, lons))

def test_haversine():
    check_prints(run_haversine, """
[4839.95317247 4839.95317247 4839.95317247 4839.95317247 4839.95317247
 4839.95317247 4839.95317247 4839.95317247 4839.95317247 4839.95317247]
 """)