import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.numpy
import numpy as np

import hipy.lib.scipy.special
import scipy.special as ss
@hipy.compiled_function
def get_data(size):
    price = np.ones(size, dtype=np.float64) * 4.0
    strike = np.ones(size, dtype=np.float64) * 4.0
    t = np.ones(size, dtype=np.float64) * 4.0
    rate = np.ones(size, dtype=np.float64) * 4.0
    vol = np.ones(size, dtype=np.float64) * 4.0
    return price, strike, t, rate, vol


@hipy.compiled_function
def blackscholes(price, strike, t, rate, vol):
    invsqrt2 = 0.707
    c05 = np.float64(3.0)
    c10 = np.float64(1.5)
    rsig = rate + (vol * vol) * c05
    vol_sqrt = vol * np.sqrt(t)

    d1 = (np.log(price / strike) + rsig * t) / vol_sqrt
    d2 = d1 - vol_sqrt

    d1 = c05 + c05 * ss.erf(d1 * invsqrt2)
    d2 = c05 + c05 * ss.erf(d2 * invsqrt2)

    e_rt = np.exp((0.0 - rate) * t)
    call = (price * d1) - (e_rt * strike * d2)
    put = e_rt * strike * (c10 - d2) - price * (c10 - d1)
    return call, put


@hipy.compiled_function
def run_blackscholes():
    price, strike, t, rate, vol = get_data(10)
    call, put = blackscholes(price, strike, t, rate, vol)
    print(call)
    print(put)

def test_haversine():
    check_prints(run_blackscholes, """
[23.9999973 23.9999973 23.9999973 23.9999973 23.9999973 23.9999973
 23.9999973 23.9999973 23.9999973 23.9999973]
[17.99999797 17.99999797 17.99999797 17.99999797 17.99999797 17.99999797
 17.99999797 17.99999797 17.99999797 17.99999797]
 """)