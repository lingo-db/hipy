import random
import re

import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import hipy.lib.pandas
import pandas as pd
import numpy as np
import hipy.lib.numpy

from hipy.value import raw_module

_pd = raw_module(pd)


@hipy.compiled_function
def gen_data(size):
    total_population = np.ones(size) * 500000
    adult_population = np.ones(size) * 250000
    num_robberies = np.ones(size) * 1000
    return pd.DataFrame.from_dict({
        "Total population": total_population,
        "Total adult population": adult_population,
        "Number of robberies": num_robberies,
    })


@hipy.compiled_function
def crime_index_weld(data):
    total_population = data["Total population"]
    adult_population = data["Total adult population"]
    num_robberies = data["Number of robberies"]
    big_cities = total_population > 500000
    big_cities = total_population.mask(big_cities, np.float64(0.0))
    double_pop = ((adult_population * 2.0) + big_cities).sub(num_robberies * 2000.0)
    crime_index = double_pop / 100000.0
    crime_index = crime_index.mask(crime_index > 0.02, 0.032)
    crime_index = crime_index.mask(crime_index < 0.01, 0.005)
    return crime_index.sum()


@hipy.compiled_function
def fn_crime_index():
    data = gen_data(10)
    print(crime_index_weld(data))


def test_crime_index():
    check_prints(fn_crime_index, """0.05""")
