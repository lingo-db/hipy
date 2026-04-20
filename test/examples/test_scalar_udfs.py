"""Tests for UDFs from the scalar-hipy benchmark
(`udf-benchmark/benchmarks/lingodb/scalar-hipy/initialize.sql`).

`check_iban` and `levenshtein_dist` are already covered by
`test_check_iban.py` and `test_levenshtein.py`. This file adds coverage
for the remaining UDFs (`parse_url_host`, `predict`).
"""
import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant

import hipy.lib.urllib.parse
from urllib.parse import urlparse

import hipy.lib.sklearn.linear_model
import hipy.lib.pickle
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np


@hipy.compiled_function
def parse_url_host(string):
    return urlparse(string).hostname


@hipy.compiled_function
def fn_parse_url_host():
    print(parse_url_host(not_constant("http://docs.python.org:80/3/library")))
    print(parse_url_host(not_constant("https://example.com/path")))


def test_parse_url_host():
    check_prints(fn_parse_url_host, """
docs.python.org
example.com
""")


# `predict` UDF body — offline training is performed once at import-time,
# exactly as the SQL UDF does in the Python preamble. The trained model
# is pickled into a `hipy.global_const` so the compiled_function can load
# it back at compile-time.
_X = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 2], [1, 2, 3]])
_y = np.dot(_X, np.array([1, 2, 3])) + 3
_reg = LinearRegression().fit(_X, _y)
_pickled = pickle.dumps(_reg)
x = hipy.global_const(_pickled)


@hipy.compiled_function
def predict(a, b, c):
    model = pickle.loads(x)
    return model.predict([[a, b, c]])[0]


@hipy.compiled_function
def fn_predict():
    print(round(predict(not_constant(3), not_constant(5), not_constant(7))))


def test_predict():
    # First column of X is constant 1, so the fit collapses it into the
    # intercept: coef ~ [0, 2, 3], intercept ~ 4 → 0*3 + 2*5 + 3*7 + 4 = 35.
    check_prints(fn_predict, "35\n")
