from sklearn.linear_model import LinearRegression
import hipy.lib.sklearn.linear_model
import pickle
import hipy.lib.pickle

import numpy as np

from hipy.interpreter import check_prints

#offline learning
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
pickled=pickle.dumps(reg)
x=hipy.global_const(pickled)


@hipy.compiled_function
def fn_inference():
    model=pickle.loads(x)
    l=model.predict([[3, 5]])
    print(round(l[0]))

def test_inference():
    check_prints(fn_inference, "16\n")

